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
        self.dataset_name = 'combined'  # 数据集名称: 'casme2', 'samm', 'combined'
        self.dataset_roots = {
            'casme2': 'CASME2/CASME2_Cropped',
            'samm': 'SAMM/SAMM_Cropped',
            'combined': 'COMBINED/COMBINED_Cropped'
        }
        
        # 自动获取当前数据集的根目录
        if self.dataset_name not in self.dataset_roots:
            raise ValueError(f"数据集 '{self.dataset_name}' 未在 dataset_roots 中配置路径！")
        
        self.root_dir = self.dataset_roots[self.dataset_name]
        self.num_frames = 16          # 每个视频采样的帧数
        self.height = 112             # 帧高度
        self.width = 112              # 帧宽度
        self.num_classes = 3          # 分类数量
        self.frame_step = 1           # 跳帧采样步长（1表示不跳帧，2表示每隔1帧取一帧）
        
        # =============================================================================
        # 训练配置
        # =============================================================================
        self.batch_size = 8             # 批次大小
        self.num_epochs = 50            # 训练轮数
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
        self.focal_alpha = []                   # Focal Loss的alpha参数
        self.focal_gamma = 1.5                  # Focal Loss的gamma参数
        self.label_smoothing = 0.0              # 与FocalLoss配合时不使用Label Smoothing
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
        self.dropout_rate = 0.3         # Dropout概率
        self.use_batch_norm = True      # 是否使用批量归一化
        self.grad_clip_norm = 1.0       # 梯度裁剪阈值 
        self.early_stopping_patience = 999 # 早停耐心值 (轮数内无提升则停止)
        self.late_select_epochs = 5     # 仅最后N轮参与最佳模型选择，避免前期震荡误选
        self.use_mixup = False           # 是否使用Mixup增强
        self.mixup_alpha = 0.1          # Mixup的Beta分布alpha参数
        
        # =============================================================================
        # 数据增强配置
        # =============================================================================
        self.use_data_augmentation = True        # 是否使用数据增强
        self.random_crop = True                  # 是否使用随机裁剪
        self.crop_size = 104                     # 裁剪大小
        self.random_scale = True                 # 是否使用随机缩放
        self.random_rotation = True              # 是否使用随机旋转
        self.random_horizontal_flip = True       # 是否使用随机水平翻转
        self.random_brightness = True            # 是否使用随机亮度增强
        self.random_contrast = False              # 是否使用随机对比度增强
        self.random_frame_dropout = True         # 是否使用随机帧丢弃
        self.frame_dropout_prob = 0.3            # 帧丢弃概率
        self.random_temporal_shuffle = True      # 是否使用时序局部打乱
        self.temporal_shuffle_prob = 0.3         # 时序打乱概率


        
        # =============================================================================
        # 光流和视频处理配置
        # =============================================================================
        self.optical_flow_type = 'tv_l1'     # 光流类型: 'farneback', 'tv_l1'
        self.video_magnification = True     # 是否使用视频放大
        self.use_two_stream = True          # 是否使用双流法
        
        # =============================================================================
        # 学习率调度器配置
        # =============================================================================
        self.use_warmup = True                      # 是否使用学习率warmup
        self.warmup_epochs = 5                    # warmup的轮数
        self.warmup_start_lr = self.learning_rate * 0.1  # warmup的起始学习率
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
        self.model_name = 'resnet3d18'  # 模型名称: 'resnet3d18', 'resnet3d34'
        self.use_attention = True       # 是否使用注意力机制
        self.attention_type = 'cbam'    # 注意力类型: 'cbam', 'self'
        
        # =============================================================================
        # 输出配置
        # =============================================================================
        self.output_dir = 'outputs'  # 输出目录
        self.dataset_output_dir = os.path.join(self.output_dir, self.dataset_name, "test")  # 数据集特定输出目录
        self.checkpoint_dir = os.path.join(self.dataset_output_dir, 'checkpoints')  # 检查点目录
        self.figure_dir = os.path.join(self.dataset_output_dir, 'figures')  # 图表目录
        self.log_dir = os.path.join(self.dataset_output_dir, 'logs')  # 日志目录
        self.log_file = os.path.join(self.log_dir, 'training.log')  # 日志文件
        self.save_checkpoint_freq = 25  # 检查点保存频率
        
        # =============================================================================
        # 其他配置
        # =============================================================================
        self.seed = 123456  # 随机种子
        self.deterministic = False  # 完全可复现模式
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备
        
        # =============================================================================
        # 数据集特定参数
        # =============================================================================
        self._set_dataset_specific_params()
    
    def _set_dataset_specific_params(self):
        """设置数据集特定的参数"""
        if self.dataset_name == 'casme2':
            # CASME2 数据集特定参数
            self.focal_alpha = [1.3205, 1.4525, 0.227]
            self.scale_range = [0.95, 1.05]          # 缩放范围
            self.rotation_range = [-3, 3]            # 旋转角度范围
            self.brightness_range = [0.90, 1.10]       # 亮度调整范围
            self.contrast_range = [0.90, 1.10]        # 对比度调整范围
            self.evm_amplification = 10.0             # 视频放大倍数
            self.evm_frequency_band = [40.0, 200.0]     # 视频放大的频率带（毫秒）
            self.fps = 200                           # 视频帧率（Hz）
        elif self.dataset_name == 'samm':
            # SAMM 数据集特定参数
            self.focal_alpha = [0.9898, 1.7817, 0.2284]
            self.scale_range = [0.95, 1.05]          # 缩放范围
            self.rotation_range = [-3, 3]            # 旋转角度范围
            self.brightness_range = [0.90, 1.10]       # 亮度调整范围
            self.contrast_range = [0.90, 1.10]        # 对比度调整范围
            self.evm_amplification = 10.0             # 视频放大倍数
            self.evm_frequency_band = [40.0, 200.0]     # 视频放大的频率带（毫秒）
            self.fps = 200                           # 视频帧率（Hz）
        else:
            # 默认参数
            self.scale_range = [0.95, 1.05]          # 缩放范围
            self.rotation_range = [-3, 3]            # 旋转角度范围
            self.brightness_range = [0.90, 1.10]       # 亮度调整范围
            self.contrast_range = [0.90, 1.10]        # 对比度调整范围
            self.evm_amplification = 10.0             # 视频放大倍数
            self.evm_frequency_band = [40.0, 200.0]     # 视频放大的频率带（毫秒）
            self.fps = 200                           # 视频帧率（Hz）