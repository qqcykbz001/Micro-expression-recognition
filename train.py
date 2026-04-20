import os
import random
import pandas as pd
import numpy as np

# 设置环境变量以解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.resnet3d import resnet3d18, resnet3d34, resnet3d50
from src.datasets import get_dataset
from src.utils.train_utils import train, test, FocalLoss
from src.utils.visualization_utils import plot_training_metrics, plot_confusion_matrix
import time
import datetime

# 导入配置
from src.configs.config import Config

# 初始化配置
config = Config()

# 使用时间戳生成唯一的日志文件名，以便区分不同次的训练
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(config.log_dir, f"training_{config.dataset_name}_{timestamp}.log")


# 日志写入函数
def log(message, level="INFO"):
    """将日志写入文件并打印到控制台"""
    # 确保日志目录存在
    log_dir = os.path.dirname(LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 获取当前时间
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 格式化消息
    formatted_message = f"[{now}] [{level}] {message}"
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted_message + "\n")
    
    # 控制台彩色输出（可选，这里使用简单的文本标记）
    print(formatted_message)

def log_config(config):
    """打印并记录当前配置参数"""
    log("-" * 30 + " 训练配置参数 " + "-" * 30)
    # 获取Config类的所有属性
    attrs = vars(config)
    for key, value in attrs.items():
        log(f"{key:.<40} {value}")
    log("-" * 74)

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 主函数
def main():
    import numpy as np
    # 设置设备
    device = config.device
    log(f'开始训练任务，数据集: {config.dataset_name}，日志文件: {LOG_FILE}', level="SUCCESS")
    log(f'使用设备: {device}')
    
    # 打印配置
    log_config(config)

    set_seed(config.seed, deterministic=config.deterministic)

    # 创建输出相关的所有目录
    model_dir = config.checkpoint_dir
    figure_dir = config.figure_dir
    for d in [config.output_dir, model_dir, figure_dir, config.log_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            log(f'创建目录: {d}')
        else:
            log(f'使用现有目录: {d}')

    # 超参数
    num_epochs = config.num_epochs
    batch_size = config.batch_size  
    learning_rate = config.learning_rate
    num_classes = config.num_classes  
    num_frames = config.num_frames 
    height, width = config.height, config.width 

    # 获取所有受试者列表
    root_dir = config.root_dir
    subjects = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    log(f'发现 {len(subjects)} 个受试者: {subjects}')

    # 存储每次验证的准确率、UAR和UF1
    accuracies = []
    uar_scores = []
    uf1_scores = []
    valid_subjects = []
    
    # 用于汇总所有fold的预测结果
    all_targets = []
    all_predicted = []

    # 执行LOSO交叉验证
    for i, test_subject in enumerate(subjects):
        log(f'LOSO Fold {i + 1}/{len(subjects)}, 测试受试者: {test_subject}')

        # 创建训练集和测试集
        exclude_subjects = [test_subject]
        log(f'创建训练集，排除受试者: {exclude_subjects}')
        train_dataset = get_dataset(config, exclude_subjects=exclude_subjects, log_func=log)
        log(f'创建测试集，包含受试者: {exclude_subjects}')
        test_dataset = get_dataset(config, include_subjects=exclude_subjects, log_func=log)

        log(f'训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}')

        # 检查测试集是否为空
        if len(test_dataset) == 0:
            log(f'警告: 测试集为空，跳过该fold')
            continue

        generator = torch.Generator()
        generator.manual_seed(config.seed + i)

        # 创建数据加载器 (启用 pin_memory 配合 non_blocking 提升性能)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            worker_init_fn=seed_worker if config.num_workers > 0 else None,
            generator=generator
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            worker_init_fn=seed_worker if config.num_workers > 0 else None,
            generator=generator
        )

        # 根据是否使用双流法设置输入通道数（单通道灰度图）
        input_channels = 4 if config.use_two_stream else 1
        
        # 创建模型
        if config.model_name == 'resnet3d18':
            log(f'创建模型: 3D ResNet-18')
            model = resnet3d18(
                num_classes=num_classes, 
                pretrained=config.pretrained,
                use_attention=config.use_attention,
                attention_type=config.attention_type,
                use_dropout=config.use_dropout,
                dropout_rate=config.dropout_rate,
                input_channels=input_channels,
                use_batch_norm=config.use_batch_norm
            ).to(device)
        elif config.model_name == 'resnet3d34':
            log(f'创建模型: 3D ResNet-34')
            model = resnet3d34(
                num_classes=num_classes, 
                pretrained=config.pretrained,
                use_attention=config.use_attention,
                attention_type=config.attention_type,
                use_dropout=config.use_dropout,
                dropout_rate=config.dropout_rate,
                input_channels=input_channels,
                use_batch_norm=config.use_batch_norm
            ).to(device)
        elif config.model_name == 'resnet3d50':
            log(f'创建模型: 3D ResNet-50')
            model = resnet3d50(
                num_classes=num_classes, 
                pretrained=config.pretrained,
                use_attention=config.use_attention,
                attention_type=config.attention_type,
                use_dropout=config.use_dropout,
                dropout_rate=config.dropout_rate,
                input_channels=input_channels,
                use_batch_norm=config.use_batch_norm
            ).to(device)
        else:
            log(f'模型名称 {config.model_name} 不支持，使用默认模型: 3D ResNet-50')
            model = resnet3d50(
                num_classes=num_classes, 
                pretrained=config.pretrained,
                use_attention=config.use_attention,
                attention_type=config.attention_type,
                use_dropout=config.use_dropout,
                dropout_rate=config.dropout_rate,
                input_channels=input_channels,
                use_batch_norm=config.use_batch_norm
            ).to(device)
        log(f'注意力配置: use_attention={config.use_attention}, attention_type={config.attention_type}')
        log(f'正则化配置: dropout={config.use_dropout}, dropout_rate={config.dropout_rate}, batch_norm={config.use_batch_norm}')
        log(f'参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

        # 定义损失函数和优化器
        if config.loss_name == 'focal':
            # 动态计算focal_alpha
            if config.use_dynamic_alpha:
                # 从训练数据集中获取类别分布
                labels = [sample['label'] for sample in train_dataset.samples]
                from collections import Counter
                counts = Counter(labels)
                total = len(labels)
                # 计算每个类别的权重
                weights = [total / counts[i] if i in counts else 1.0 for i in range(config.num_classes)]
                # 归一化权重
                mean_w = sum(weights) / len(weights)
                focal_alpha = [w / mean_w for w in weights]
                log(f'使用动态计算的 focal_alpha: {[round(a, 3) for a in focal_alpha]}')
            else:
                focal_alpha = config.focal_alpha
                log(f'使用静态 focal_alpha: {focal_alpha}')
            log(f'定义损失函数: FocalLoss(alpha={[round(a, 3) for a in focal_alpha]}, gamma={config.focal_gamma}, smoothing={config.label_smoothing})')
            criterion = FocalLoss(alpha=focal_alpha, gamma=config.focal_gamma, label_smoothing=config.label_smoothing)
        elif config.loss_name == 'cross_entropy':
            log(f'定义损失函数: CrossEntropyLoss(smoothing={config.label_smoothing})')
            criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        else:
            log(f'损失函数名称 {config.loss_name} 不支持，使用默认损失函数: FocalLoss')
            # 动态计算focal_alpha
            if config.use_dynamic_alpha:
                # 从训练数据集中获取类别分布
                labels = [sample['label'] for sample in train_dataset.samples]
                from collections import Counter
                counts = Counter(labels)
                total = len(labels)
                # 计算每个类别的权重
                weights = [total / counts[i] if i in counts else 1.0 for i in range(config.num_classes)]
                # 归一化权重
                mean_w = sum(weights) / len(weights)
                focal_alpha = [w / mean_w for w in weights]
                log(f'使用动态计算的 focal_alpha: {[round(a, 3) for a in focal_alpha]}')
            else:
                focal_alpha = config.focal_alpha
                log(f'使用静态 focal_alpha: {focal_alpha}')
            criterion = FocalLoss(alpha=focal_alpha, gamma=config.focal_gamma, label_smoothing=config.label_smoothing)
        # 定义优化器
        if config.optimizer_name == 'sgd':
            log(f'定义优化器: SGD(lr={learning_rate}, momentum={config.sgd_momentum}, weight_decay={config.weight_decay})')
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=config.sgd_momentum, weight_decay=config.weight_decay)
        elif config.optimizer_name == 'adamw':
            log(f'定义优化器: AdamW(lr={learning_rate}, beta1={config.adamw_beta1}, beta2={config.adamw_beta2}, eps={config.adamw_eps}, weight_decay={config.weight_decay})')
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(config.adamw_beta1, config.adamw_beta2), eps=config.adamw_eps, weight_decay=config.weight_decay)
        else:
            log(f'优化器名称 {config.optimizer_name} 不支持，使用默认优化器: AdamW')
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(config.adamw_beta1, config.adamw_beta2), eps=config.adamw_eps, weight_decay=config.weight_decay)

        # 添加学习率调度器
        if config.scheduler_name == 'cosine':
            log(f'定义学习率调度器: CosineAnnealingLR(T_max={config.cosine_t_max})')
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.cosine_t_max)
        elif config.scheduler_name == 'step':
            log(f'定义学习率调度器: StepLR(step_size={config.step_size}, gamma={config.gamma})')
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
        elif config.scheduler_name == 'reduce_lr_on_plateau':
            log(f'定义学习率调度器: ReduceLROnPlateau(factor={config.gamma}, patience=5)')
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.gamma, patience=5)
        else:
            log(f'调度器名称 {config.scheduler_name} 不支持，使用默认调度器: CosineAnnealingLR')
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.cosine_t_max)
        
        # 学习率warmup相关参数
        use_warmup = getattr(config, 'use_warmup', False)
        warmup_epochs = getattr(config, 'warmup_epochs', 5)
        warmup_start_lr = getattr(config, 'warmup_start_lr', 1e-6)
        
        if use_warmup:
            log(f'启用学习率warmup: {warmup_epochs}轮, 起始学习率: {warmup_start_lr}')
            # 计算warmup的学习率增长步长
            warmup_step = (learning_rate - warmup_start_lr) / warmup_epochs

        # 检查点保存路径
        checkpoint_path = os.path.join(model_dir, f'checkpoint_{config.dataset_name}_fold{i + 1}.pth')

        # 用于可视化的记录
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        test_uar_scores = []
        test_uf1s = []
        learning_rates = []

        # 尝试加载检查点
        start_epoch = 0
        best_accuracy = 0.0

        if os.path.exists(checkpoint_path):
            log(f'从 {checkpoint_path} 加载检查点...')
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_accuracy = checkpoint['best_accuracy']
            train_losses = checkpoint['train_losses']
            train_accuracies = checkpoint['train_accuracies']
            test_accuracies = checkpoint['test_accuracies']
            test_uar_scores = checkpoint.get('test_uar_scores', [])
            test_uf1s = checkpoint.get('test_uf1s', [])
            learning_rates = checkpoint['learning_rates']
            log(f'检查点加载成功。从第 {start_epoch} 轮开始训练')
        else:
            log('未找到检查点。从头开始训练。')
            log('开始训练...')

        for epoch in range(start_epoch, num_epochs):
            # 学习率warmup
            if use_warmup and epoch < warmup_epochs:
                # 计算当前warmup学习率
                current_lr = warmup_start_lr + epoch * warmup_step
                # 设置学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                log(f'Epoch {epoch + 1}/{num_epochs}, 学习率 (warmup): {current_lr:.6f}')
            else:
                current_lr = optimizer.param_groups[0]["lr"]
                log(f'Epoch {epoch + 1}/{num_epochs}, 学习率: {current_lr:.6f}')
            # 记录epoch开始时间
            epoch_start = time.time()

            # 训练模型
            log(f'开始训练轮次 {epoch + 1}/{num_epochs}...')
            train_loss, train_acc = train(
                model, train_loader, criterion, optimizer, device,
                accumulation_steps=config.accumulation_steps,
                use_amp=config.use_amp,
                log_func=log
            )
            log(f'Epoch {epoch + 1} 训练完成: Loss={train_loss:.4f}, Acc={train_acc:.2f}%', level="SUCCESS")

            # 测试模型
            log(f'开始测试轮次 {epoch + 1}...')
            test_acc, test_uar, test_uf1, fold_targets, fold_predicted = test(model, test_loader, criterion, device, log_func=log)
            log(f'Epoch {epoch + 1} 测试完成: Acc={test_acc:.2f}%, UAR={test_uar:.2f}%, UF1={test_uf1:.2f}%', level="SUCCESS")

            # 记录数据用于可视化
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            test_uar_scores.append(test_uar)
            test_uf1s.append(test_uf1)
            learning_rates.append(current_lr)

            # 更新学习率 - 确保在optimizer.step()之后调用
            if use_warmup and epoch < warmup_epochs:
                # warmup阶段不更新调度器
                pass
            else:
                if config.scheduler_name == 'reduce_lr_on_plateau':
                    # ReduceLROnPlateau需要传入验证损失
                    scheduler.step(train_loss)
                else:
                    # 其他调度器直接调用
                    scheduler.step()

            # 计算epoch时间
            epoch_time = time.time() - epoch_start
            log(f'Epoch {epoch + 1} completed in {epoch_time:.2f} seconds')

            # 保存最佳模型 - 基于准确率 (在单受试者测试集上比UAR更稳定)
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_model_path = os.path.join(model_dir, f'best_resnet3d_model_{config.dataset_name}_fold{i + 1}.pth')
                torch.save(model.state_dict(), best_model_path)
                log(f'✓ 最佳模型已保存到 {best_model_path}，准确率: {best_accuracy:.2f}%')
            else:
                log(f'当前最佳准确率: {best_accuracy:.2f}%')

            # 根据频率保存检查点
            if (epoch + 1) % config.save_checkpoint_freq == 0 or (epoch + 1) == num_epochs:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_accuracy': best_accuracy,
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'test_accuracies': test_accuracies,
                    'test_uar_scores': test_uar_scores,
                    'test_uf1s': test_uf1s,
                    'learning_rates': learning_rates
                }
                torch.save(checkpoint, checkpoint_path)
                log(f'检查点已保存到 {checkpoint_path}')

            # 打印当前轮次的详细信息
            log(f'Epoch {epoch + 1} 详细信息:')
            log(f'  训练损失: {train_loss:.4f}')
            log(f'  训练准确率: {train_acc:.2f}%')
            log(f'  测试准确率: {test_acc:.2f}%')
            log(f'  测试UF1: {test_uf1:.2f}%')
            log(f'  学习率: {current_lr:.6f}')
            log(f'  耗时: {epoch_time:.2f}秒')

        # 生成可视化图表
        plot_training_metrics(train_losses, train_accuracies, test_accuracies, test_uar_scores, test_uf1s, learning_rates, i + 1,
                              config.figure_dir, dataset_name=config.dataset_name)

        # 计算最佳UF1和UAR分数
        if test_uf1s:
            best_uf1 = max(test_uf1s)
        else:
            best_uf1 = 0.0
        if test_uar_scores:
            best_uar = max(test_uar_scores)
        else:
            best_uar = 0.0

        # 收集该fold的最佳模型的预测结果
        # 重新加载最佳模型并测试
        best_model_path = os.path.join(model_dir, f'best_resnet3d_model_{config.dataset_name}_fold{i + 1}.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            # 测试最佳模型并收集结果
            _, _, _, fold_targets, fold_predicted = test(model, test_loader, criterion, device, log_func=log)
            # 添加到全局列表
            all_targets.extend(fold_targets)
            all_predicted.extend(fold_predicted)

        accuracies.append(best_accuracy)
        uar_scores.append(best_uar)
        uf1_scores.append(best_uf1)
        valid_subjects.append(test_subject)
        log(f'epoch {i + 1} 完成 最佳Acc: {best_accuracy:.3f}%, 最佳 UAR: {best_uar:.3f}%, 最佳 UF1: {best_uf1:.3f}%')

    if accuracies:
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        avg_uar = np.mean(uar_scores)
        std_uar = np.std(uar_scores)
        avg_uf1 = np.mean(uf1_scores)
        std_uf1 = np.std(uf1_scores)

        log("=" * 30 + f" LOSO 交叉验证最终总结 ({config.dataset_name}) " + "=" * 30, level="SUCCESS")
        log(f"总计 Fold 数: {len(accuracies)}")
        log(f"平均准确率 (Mean Acc): {avg_acc:.3f}% (±{std_acc:.3f}%)")
        log(f"平均 UAR (Mean UAR): {avg_uar:.3f}% (±{std_uar:.3f}%)")
        log(f"平均 UF1 分数 (Mean UF1): {avg_uf1:.3f}% (±{std_uf1:.3f}%)")
        log("-" * 74)
        
        # 汇总评估指标
        all_targets = np.array(all_targets)
        all_predicted = np.array(all_predicted)
        
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predicted, average='macro', zero_division=0)
        overall_acc = accuracy_score(all_targets, all_predicted) * 100
        overall_uar = recall * 100
        overall_uf1 = f1 * 100
        
        log(f"汇总准确率 (Overall Acc): {overall_acc:.3f}%")
        log(f"汇总 UAR (Overall UAR): {overall_uar:.3f}%")
        log(f"汇总 UF1 (Overall UF1): {overall_uf1:.3f}%")
        log("=" * 74, level="SUCCESS")
        
        # 绘制混淆矩阵
        classes = train_dataset.get_class_names()
        plot_confusion_matrix(all_targets, all_predicted, classes, config.figure_dir, dataset_name=config.dataset_name)
        
        # 导出结果到CSV
        results_df = pd.DataFrame({
            'Fold': valid_subjects,
            'Accuracy': accuracies,
            'UAR': uar_scores,
            'UF1': uf1_scores
        })
        # 计算平均值和标准差
        summary_row = pd.DataFrame({
            'Fold': ['Average', 'Std'],
            'Accuracy': [avg_acc, std_acc],
            'UAR': [avg_uar, std_uar],
            'UF1': [avg_uf1, std_uf1]
        })
        results_df = pd.concat([results_df, summary_row], ignore_index=True)
        csv_path = os.path.join(config.dataset_output_dir, f'loso_results_{config.dataset_name}_{timestamp}.csv')
        results_df.to_csv(csv_path, index=False)
        log(f'结果已导出到 CSV: {csv_path}', level="SUCCESS")
    else:
        log("未完成任何有效 Fold 的训练。", level="WARNING")
    
    log('训练成功完成！')


if __name__ == '__main__':
    main()