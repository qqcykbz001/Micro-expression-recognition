# 训练配置

# 设置环境变量以解决OpenMP冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 导入必要的库
import torch

class Config:
    """训练配置类"""
    def __init__(self):
        # 数据集配置
        self.dataset_name = 'samm'  # 数据集名称: 'casme2', 'samm', etc.
        self.dataset_roots = {
            'casme2': 'CASME2/Cropped',
            'samm': 'SAMM/SAMM'
        }
        
        # 自动获取当前数据集的根目录
        if self.dataset_name not in self.dataset_roots:
            raise ValueError(f"数据集 '{self.dataset_name}' 未在 dataset_roots 中配置路径！")
        self.root_dir = self.dataset_roots[self.dataset_name]
        self.num_frames = 16  # 每个视频采样的帧数
        self.height = 112  # 帧高度
        self.width = 112  # 帧宽度
        self.num_classes = 3  # 分类数量
        self.frame_step = 1  # 跳帧采样步长（1表示不跳帧，2表示每隔1帧取一帧）
        
        # 训练配置
        self.batch_size = 8  # 批次大小
        self.num_epochs = 50  # 训练轮数
        self.learning_rate = 1e-4  # 学习率
        self.accumulation_steps = 2  # 梯度累积步数
        self.use_amp = True  # 是否使用混合精度训练
        self.num_workers = 4  # DataLoader 并行进程数
        self.persistent_workers = True  # 是否复用 DataLoader workers
        self.prefetch_factor = 2  # 每个 worker 预取批次数
        
        # 损失函数配置
        self.loss_name = 'focal'  # 损失函数名称: 'focal', 'cross_entropy'
        self.focal_alpha = [1.0, 2.0, 0.8]  # Focal Loss的alpha参数
        self.focal_gamma = 2.0  # Focal Loss的gamma参数
        self.label_smoothing = 0.1  # 标签平滑系数 (提升泛化能力)
        
        # 优化器配置
        self.optimizer_name = 'adamw'  # 优化器名称: 'sgd', 'adamw'
        self.sgd_momentum = 0.9  # SGD的动量参数
        self.weight_decay = 1e-4  # 权重衰减参数
        self.adamw_beta1 = 0.9  # AdamW的beta1参数
        self.adamw_beta2 = 0.999  # AdamW的beta2参数
        self.adamw_eps = 1e-8  # AdamW的epsilon参数
        
        # 正则化配置
        self.use_dropout = True  # 是否使用dropout
        self.dropout_rate = 0.5  # Dropout概率
        self.use_batch_norm = True  # 是否使用批量归一化
        
        # 数据增强配置
        self.use_data_augmentation = True  # 是否使用数据增强
        self.random_crop = False  # 是否使用随机裁剪
        self.crop_size = 94  # 裁剪大小
        self.random_scale = True  # 是否使用随机缩放
        self.scale_range = [0.9, 1.1]  # 缩放范围
        self.random_rotation = False  # 是否使用随机旋转
        self.rotation_range = [-3, 3]  # 旋转角度范围
        
        # 光流特征增强配置
        self.optical_flow_type = 'tv_l1'  # 光流类型: 'farneback', 'tv_l1'
        self.use_evm = False  # 是否使用欧拉视频放大
        self.evm_amplification = 10.0  # 欧拉视频放大倍数
        self.evm_frequency_band = [0.1, 0.3]  # 欧拉视频放大的频率带
        self.use_two_stream = True  # 是否使用双流法
        
        # 学习率调度器配置
        self.scheduler_name = 'cosine'  # 调度器名称: 'cosine', 'step', 'reduce_lr_on_plateau'
        self.cosine_t_max = self.num_epochs  # CosineAnnealingLR的T_max参数
        self.step_size = 10  # StepLR的step_size参数
        self.gamma = 0.1  # StepLR的gamma参数
        
        # 模型配置
        self.model_name = 'resnet3d18'  # 模型名称: 'resnet3d18', 'resnet3d50'
        self.use_attention = True  # 是否使用注意力机制
        self.attention_type = 'cbam'  # 注意力类型: 'cbam', 'self'
        self.pretrained = False  # 是否使用预训练权重
        
        # 输出配置
        self.output_dir = 'outputs'  # 输出目录
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')  # 检查点目录
        self.figure_dir = os.path.join(self.output_dir, 'figures')  # 图表目录
        self.log_dir = os.path.join(self.output_dir, 'logs')  # 日志目录
        self.log_file = os.path.join(self.log_dir, 'training.log')  # 日志文件
        self.save_checkpoint_freq = 1  # 检查点保存频率
        
        # 其他配置
        self.seed = 131  # 随机种子
        self.deterministic = False  # 是否启用完全可复现模式（会降低训练速度）
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备