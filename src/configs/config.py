# 训练配置

# 设置环境变量以解决OpenMP冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 导入必要的库
import torch

class Config:
    """训练配置类"""
    def __init__(self):
        # =============================================================================
        # 数据集配置
        # =============================================================================
        self.dataset_name = 'casme2'  # 数据集名称: 'casme2', 'samm'
        self.dataset_roots = {
            'casme2': 'CASME2/CASME2_Cropped',
            'samm': 'SAMM/SAMM_Cropped'
        }
        
        # 自动获取当前数据集的根目录
        if self.dataset_name not in self.dataset_roots:
            raise ValueError(f"数据集 '{self.dataset_name}' 未在 dataset_roots 中配置路径！")
        
        self.root_dir = self.dataset_roots[self.dataset_name]
        self.num_frames = 16          # 每个视频采样的帧数
        self.height = 224             # 帧高度
        self.width = 224              # 帧宽度
        self.num_classes = 3          # 分类数量
        self.frame_step = 1           # 跳帧采样步长（1表示不跳帧，2表示每隔1帧取一帧）
        
        # =============================================================================
        # 训练配置
        # =============================================================================
        self.batch_size = 16            # 批次大小
        self.num_epochs = 60            # 训练轮数
        self.learning_rate = 1e-2       # 学习率
        self.accumulation_steps = 1     # 梯度累积步数
        self.use_amp = True             # 是否使用混合精度训练
        self.num_workers = 4            # DataLoader 并行进程数
        self.persistent_workers = True  # 是否复用 DataLoader workers
        self.prefetch_factor = 2        # 每个 worker 预取批次数
        
        # =============================================================================
        # 损失函数配置
        # =============================================================================
        self.loss_name = 'focal'                # 损失函数名称: 'focal', 'cross_entropy'
        self.focal_alpha = [1.0, 1.0, 1.0]     # Focal Loss的alpha参数
        self.focal_gamma = 1.5                  # Focal Loss的gamma参数
        self.label_smoothing = 0.1              # 标签平滑系数 (提升泛化能力)
        self.use_dynamic_alpha = True           # 是否使用动态计算的alpha值
        
        # =============================================================================
        # 优化器配置
        # =============================================================================
        self.optimizer_name = 'sgd'     # 优化器名称: 'sgd', 'adamw'
        self.sgd_momentum = 0.9         # SGD的动量参数
        self.weight_decay = 1e-3        # 权重衰减参数
        self.adamw_beta1 = 0.9          # AdamW的beta1参数
        self.adamw_beta2 = 0.999        # AdamW的beta2参数
        self.adamw_eps = 1e-8           # AdamW的epsilon参数
        
        # =============================================================================
        # 正则化配置
        # =============================================================================
        self.use_dropout = True         # 是否使用dropout
        self.dropout_rate = 0.5         # Dropout概率
        self.use_batch_norm = True      # 是否使用批量归一化
        
        # =============================================================================
        # 数据增强配置
        # =============================================================================
        self.use_data_augmentation = True        # 是否使用数据增强
        self.random_crop = True                  # 是否使用随机裁剪
        self.crop_size = 214                     # 裁剪大小
        self.random_scale = True                 # 是否使用随机缩放
        self.scale_range = [0.95, 1.05]            # 缩放范围
        self.random_rotation = True              # 是否使用随机旋转
        self.rotation_range = [-3, 3]            # 旋转角度范围
        self.random_horizontal_flip = True       # 是否使用随机水平翻转
        self.random_brightness = True            # 是否使用随机亮度增强
        self.brightness_range = [0.8, 1.2]       # 亮度调整范围
        self.random_contrast = True              # 是否使用随机对比度增强
        self.contrast_range = [0.8, 1.2]         # 对比度调整范围
        
        # =============================================================================
        # 光流和视频处理配置
        # =============================================================================
        self.optical_flow_type = 'tv_l1'     # 光流类型: 'farneback', 'tv_l1'
        self.video_magnification = True     # 是否使用视频放大
        self.evm_amplification = 10.0        # 视频放大倍数
        self.evm_frequency_band = [2.0, 10.0] # 视频放大的频率带（Hz）
        self.fps = 200                       # 视频帧率（Hz）
        self.use_two_stream = True          # 是否使用双流法
        self.flow_weight_initial = 0.5      # 双流法光流权重初始值
        
        # =============================================================================
        # 学习率调度器配置
        # =============================================================================
        self.use_warmup = True                      # 是否使用学习率warmup
        self.warmup_epochs = 10                    # warmup的轮数
        self.warmup_start_lr = self.learning_rate / self.warmup_epochs  # warmup的起始学习率
        self.scheduler_name = 'cosine'             # 调度器名称: 'cosine', 'step', 'reduce_lr_on_plateau'
        
        if self.use_warmup:
            self.cosine_t_max = self.num_epochs - self.warmup_epochs  # CosineAnnealingLR的T_max参数
        else:
            self.cosine_t_max = self.num_epochs  # CosineAnnealingLR的T_max参数
            
        self.step_size = 10  # StepLR的step_size参数
        self.gamma = 0.5     # StepLR的gamma参数
        
        # =============================================================================
        # 模型配置
        # =============================================================================
        self.model_name = 'resnet3d18'  # 模型名称: 'resnet3d18', 'resnet3d34', 'resnet3d50'
        self.use_attention = True       # 是否使用注意力机制
        self.attention_type = 'cbam'    # 注意力类型: 'cbam', 'self'
        self.pretrained = False          # 是否使用预训练权重
        
        # =============================================================================
        # 输出配置
        # =============================================================================
        self.output_dir = 'outputs'  # 输出目录
        self.dataset_output_dir = os.path.join(self.output_dir, self.dataset_name)  # 数据集特定输出目录
        self.checkpoint_dir = os.path.join(self.dataset_output_dir, 'checkpoints')  # 检查点目录
        self.figure_dir = os.path.join(self.dataset_output_dir, 'figures')  # 图表目录
        self.log_dir = os.path.join(self.dataset_output_dir, 'logs')  # 日志目录
        self.log_file = os.path.join(self.log_dir, 'training.log')  # 日志文件
        self.save_checkpoint_freq = 10  # 检查点保存频率
        
        # =============================================================================
        # 其他配置
        # =============================================================================
        self.seed = 42  # 随机种子
        self.deterministic = False  # 是否启用完全可复现模式（会降低训练速度）
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备