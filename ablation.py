"""消融实验：baseline / +双流 / +CBAM / +EVM / all"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import copy
import time
import datetime
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.configs.config import Config
from src.datasets import get_dataset
from src.models.resnet3d import resnet3d18
from src.utils.train_utils import train, test, FocalLoss
from src.utils.visualization_utils import plot_training_metrics, plot_confusion_matrix, plot_loso_summary

# ---------------------------------------------------------------------------
ABLATIONS = {
    "baseline":    {"use_attention": False, "use_two_stream": False, "video_magnification": False},
    "+two_stream": {"use_attention": False, "use_two_stream": True,  "video_magnification": False},
    "+cbam":       {"use_attention": True,  "use_two_stream": False, "video_magnification": False},
    "+evm":        {"use_attention": False, "use_two_stream": False, "video_magnification": True},
    "all":         {"use_attention": True,  "use_two_stream": True,  "video_magnification": True},
}

EXP_ORDER = list(ABLATIONS.keys())


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
def _noop(*args, **kwargs):
    pass


def run_experiment(config, ds_name, exp_name):
    """运行完整 LOSO，返回汇总指标（与 train.py main() 保持一致）"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = os.path.join(config.log_dir, f"training_{config.dataset_name}_{timestamp}.log")

    def _log(message, level="INFO"):
        log_dir = os.path.dirname(LOG_FILE)
        os.makedirs(log_dir, exist_ok=True)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{now}] [{level}] {message}"
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
        print(formatted)

    def _log_config(cfg):
        _log("-" * 30 + " 训练配置参数 " + "-" * 30)
        for key, value in vars(cfg).items():
            _log(f"{key:.<40} {value}")
        _log("-" * 74)

    device = config.device
    _log(f'=== {exp_name} | {ds_name} | 设备: {device} | 种子: {config.seed}', level="SUCCESS")
    _log_config(config)
    set_seed(config.seed, deterministic=config.deterministic)

    # 创建目录
    for d in [config.output_dir, config.checkpoint_dir, config.figure_dir, config.log_dir]:
        os.makedirs(d, exist_ok=True)

    root_dir = config.root_dir
    subjects = sorted([s for s in os.listdir(root_dir)
                       if os.path.isdir(os.path.join(root_dir, s))])
    _log(f'发现 {len(subjects)} 个受试者: {subjects}')

    accuracies, uar_scores, uf1_scores = [], [], []
    valid_subjects = []
    all_targets, all_predicted = [], []

    for i, test_subject in enumerate(subjects):
        _log(f'LOSO Fold {i + 1}/{len(subjects)}, 测试受试者: {test_subject}')

        train_dataset = get_dataset(config, exclude_subjects=[test_subject], log_func=print)
        test_cfg = copy.deepcopy(config)
        test_cfg.use_data_augmentation = False
        test_dataset = get_dataset(test_cfg, include_subjects=[test_subject], log_func=print)
        _log(f'训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}')

        if len(test_dataset) == 0:
            _log('警告: 测试集为空，跳过该fold')
            continue

        # DataLoader
        generator = torch.Generator()
        generator.manual_seed(config.seed + i)

        train_labels = [s['label'] for s in train_dataset.samples]
        class_counts = {l: train_labels.count(l) for l in set(train_labels)}
        sampler = WeightedRandomSampler(
            [1.0 / class_counts[l] for l in train_labels],
            num_samples=len(train_labels), replacement=True)

        nw = config.num_workers
        dl_kw = dict(pin_memory=True, num_workers=nw,
                     persistent_workers=config.persistent_workers if nw > 0 else False,
                     prefetch_factor=config.prefetch_factor if nw > 0 else None,
                     worker_init_fn=seed_worker if nw > 0 else None)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  sampler=sampler, generator=generator, **dl_kw)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                                 shuffle=False, generator=generator, **dl_kw)

        # 模型
        in_ch = 4 if config.use_two_stream else 3
        model = resnet3d18(
            num_classes=config.num_classes, use_attention=config.use_attention,
            attention_type=config.attention_type, use_dropout=config.use_dropout,
            dropout_rate=config.dropout_rate, input_channels=in_ch,
            use_batch_norm=config.use_batch_norm, config=config,
        ).to(device)
        _log(f'模型: ResNet3D-18 | 参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} | '
             f'attention={config.use_attention} two_stream={config.use_two_stream} evm={config.video_magnification}')

        # 损失 (动态 focal_alpha)
        if config.loss_name == 'focal':
            labels = [s['label'] for s in train_dataset.samples]
            counts = Counter(labels)
            total = len(labels)
            weights = [total / counts[i] if i in counts else 1.0 for i in range(config.num_classes)]
            mean_w = sum(weights) / len(weights)
            alpha = [w / mean_w for w in weights]
            criterion = FocalLoss(alpha=alpha, gamma=config.focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss()

        # 优化器 & 调度器
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                              momentum=config.sgd_momentum, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.cosine_t_max)
        warmup_step = (config.learning_rate - config.warmup_start_lr) / config.warmup_epochs

        model_dir = config.checkpoint_dir
        checkpoint_path = os.path.join(model_dir, f'checkpoint_{config.dataset_name}_fold{i + 1}.pth')

        # 训练状态
        train_losses, train_accuracies = [], []
        test_accuracies_l, test_uar_scores, test_uf1s, learning_rates = [], [], [], []
        best_combined_score = 0.0
        best_acc, best_uar, best_uf1 = 0.0, 0.0, 0.0
        patience_counter = 0
        late_select_start = config.num_epochs - config.late_select_epochs

        _log('开始训练...')
        for epoch in range(config.num_epochs):
            # warmup
            if config.use_warmup and epoch < config.warmup_epochs:
                if epoch == config.warmup_epochs - 1:
                    current_lr = config.learning_rate
                else:
                    current_lr = config.warmup_start_lr + epoch * warmup_step
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            epoch_start = time.time()

            train_loss, train_acc = train(
                model, train_loader, criterion, optimizer, device,
                use_amp=config.use_amp, use_mixup=config.use_mixup,
                mixup_alpha=config.mixup_alpha, grad_clip_norm=config.grad_clip_norm,
                num_classes=config.num_classes, log_func=_noop)

            test_acc, test_uar, test_uf1, _, _ = test(
                model, test_loader, criterion, device, log_func=_noop)

            epoch_time = time.time() - epoch_start
            combined_score = (test_acc + test_uar + test_uf1) / 3

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_accuracies_l.append(test_acc)
            test_uar_scores.append(test_uar)
            test_uf1s.append(test_uf1)
            learning_rates.append(current_lr)

            if config.use_warmup and epoch < config.warmup_epochs:
                pass
            else:
                scheduler.step()

            phase = "WARM" if (config.use_warmup and epoch < config.warmup_epochs) else \
                    ("LATE" if epoch >= late_select_start else "TRN")
            _log(f'Epoch {epoch + 1:>3d}/{config.num_epochs} | '
                 f'TL:{train_loss:.4f} TA:{train_acc:5.1f}% | '
                 f'VA:{test_acc:5.1f}% UAR:{test_uar:5.1f}% UF1:{test_uf1:5.1f}% | '
                 f'CS:{combined_score:5.1f}% | LR:{current_lr:.2e} | {epoch_time:.1f}s [{phase}]',
                 level="SUCCESS")

            if epoch >= late_select_start:
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_acc, best_uar, best_uf1 = test_acc, test_uar, test_uf1
                    patience_counter = 0
                    best_model_path = os.path.join(model_dir, f'best_resnet3d_model_{config.dataset_name}_fold{i + 1}.pth')
                    torch.save(model.state_dict(), best_model_path)
                    _log(f'  >> 最佳已更新 | CS:{combined_score:.1f}%')
                else:
                    patience_counter += 1

                if patience_counter >= config.early_stopping_patience:
                    _log(f'早停触发 (连续{config.early_stopping_patience}轮无提升)')
                    break

            if (epoch + 1) % config.save_checkpoint_freq == 0 or (epoch + 1) == config.num_epochs:
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_combined_score': best_combined_score,
                    'best_acc': best_acc, 'best_uar': best_uar, 'best_uf1': best_uf1,
                    'patience_counter': patience_counter,
                    'train_losses': train_losses, 'train_accuracies': train_accuracies,
                    'test_accuracies': test_accuracies_l, 'test_uar_scores': test_uar_scores,
                    'test_uf1s': test_uf1s, 'learning_rates': learning_rates,
                }, checkpoint_path)
                _log(f'检查点已保存到 {checkpoint_path}')

        # 可视化
        plot_training_metrics(train_losses, train_accuracies, test_accuracies_l,
                              test_uar_scores, test_uf1s, learning_rates, i + 1,
                              config.figure_dir, dataset_name=config.dataset_name,
                              late_select_start=late_select_start)

        # 重载最佳模型，收集预测
        best_model_path = os.path.join(model_dir, f'best_resnet3d_model_{config.dataset_name}_fold{i + 1}.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
        _, _, _, fold_targets, fold_pred = test(
            model, test_loader, criterion, device, log_func=_noop)

        accuracies.append(best_acc)
        uar_scores.append(best_uar)
        uf1_scores.append(best_uf1)
        valid_subjects.append(test_subject)
        all_targets.extend(fold_targets)
        all_predicted.extend(fold_pred)

        _log(f'Fold {i + 1} 完成 最佳 Acc:{best_acc:.1f}% UAR:{best_uar:.1f}% UF1:{best_uf1:.1f}%')

    if not accuracies:
        _log("未完成任何有效 Fold。", level="WARNING")
        return None

    # 汇总
    all_targets = np.array(all_targets)
    all_predicted = np.array(all_predicted)
    _, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predicted, average='macro', zero_division=0)
    overall_acc = accuracy_score(all_targets, all_predicted) * 100
    overall_uar = recall * 100
    overall_uf1 = f1 * 100

    _log("=" * 30 + f" LOSO 最终总结 ({config.dataset_name}) " + "=" * 30, level="SUCCESS")
    _log(f"Mean Acc: {np.mean(accuracies):.2f}% (±{np.std(accuracies):.2f}%)")
    _log(f"Mean UAR: {np.mean(uar_scores):.2f}% (±{np.std(uar_scores):.2f}%)")
    _log(f"Mean UF1: {np.mean(uf1_scores):.2f}% (±{np.std(uf1_scores):.2f}%)")
    _log(f"Overall Acc: {overall_acc:.2f}%  UAR: {overall_uar:.2f}%  UF1: {overall_uf1:.2f}%")
    _log("=" * 74, level="SUCCESS")

    # 可视化: 混淆矩阵 + LOSO 汇总图
    classes = train_dataset.get_class_names()
    plot_confusion_matrix(all_targets, all_predicted, classes,
                          config.figure_dir, dataset_name=config.dataset_name)
    plot_loso_summary(accuracies, uar_scores, uf1_scores, valid_subjects,
                      config.figure_dir, dataset_name=config.dataset_name)

    # CSV
    df = pd.DataFrame({'Fold': valid_subjects, 'Accuracy': accuracies,
                       'UAR': uar_scores, 'UF1': uf1_scores})
    summary = pd.DataFrame({'Fold': ['Average', 'Std'],
                            'Accuracy': [np.mean(accuracies), np.std(accuracies)],
                            'UAR': [np.mean(uar_scores), np.std(uar_scores)],
                            'UF1': [np.mean(uf1_scores), np.std(uf1_scores)]})
    df = pd.concat([df, summary], ignore_index=True)
    csv_path = os.path.join(config.dataset_output_dir, f'loso_results_{config.dataset_name}.csv')
    df.to_csv(csv_path, index=False)
    _log(f'结果已导出到 CSV: {csv_path}', level="SUCCESS")

    return {
        "exp": exp_name, "dataset": ds_name,
        "folds": len(accuracies),
        "mean_acc": np.mean(accuracies), "std_acc": np.std(accuracies),
        "mean_uar": np.mean(uar_scores), "std_uar": np.std(uar_scores),
        "mean_uf1": np.mean(uf1_scores), "std_uf1": np.std(uf1_scores),
        "overall_acc": overall_acc, "overall_uar": overall_uar, "overall_uf1": overall_uf1,
    }


# ---------------------------------------------------------------------------
def main():
    datasets = ["casme2"]
    all_results = []

    for ds_name in datasets:
        for exp_name in EXP_ORDER:
            print(f"\n{'#' * 70}")
            print(f"#  {exp_name}  |  {ds_name}")
            print(f"{'#' * 70}")

            cfg = Config()
            cfg.dataset_name = ds_name
            cfg.root_dir = cfg.dataset_roots[ds_name]

            # 基线组件（其余全开，被消融的三项由 ABLATIONS 决定）
            cfg.use_mixup = True
            cfg.random_frame_dropout = True
            cfg.random_temporal_shuffle = True
            cfg.use_data_augmentation = True
            cfg.loss_name = "focal"
            cfg.use_dynamic_alpha = True

            # 应用消融变化
            for k, v in ABLATIONS[exp_name].items():
                setattr(cfg, k, v)

            # 输出目录按实验名区分
            cfg.dataset_output_dir = os.path.join(cfg.output_dir, cfg.dataset_name, exp_name)
            cfg.checkpoint_dir = os.path.join(cfg.dataset_output_dir, 'checkpoints')
            cfg.figure_dir = os.path.join(cfg.dataset_output_dir, 'figures')
            cfg.log_dir = os.path.join(cfg.dataset_output_dir, 'logs')
            cfg.log_file = os.path.join(cfg.log_dir, 'training.log')

            res = run_experiment(cfg, ds_name, exp_name)
            if res:
                all_results.append(res)

    # =========================================================================
    # 汇总表
    # =========================================================================
    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"ablation_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

        print("\n" + "=" * 90)
        print("消融实验结果汇总 (Overall Metrics)")
        print("=" * 90)
        header = (f"{'Experiment':<16s} "
                  f"{'CASME2 Acc':>10s} {'CASME2 UAR':>10s} {'CASME2 UF1':>10s}  "
                  f"{'SAMM Acc':>9s} {'SAMM UAR':>9s} {'SAMM UF1':>9s}")
        print(header)
        print("-" * 90)

        for exp_name in EXP_ORDER:
            row = f"{exp_name:<16s}"
            for ds in datasets:
                s = df[(df["exp"] == exp_name) & (df["dataset"] == ds)]
                if not s.empty:
                    r = s.iloc[0]
                    row += f" {r['overall_acc']:>8.2f}% {r['overall_uar']:>8.2f}% {r['overall_uf1']:>8.2f}%  "
            print(row)

        print("=" * 90)
        print(f"完整结果已保存到: {csv_path}")


if __name__ == "__main__":
    main()
