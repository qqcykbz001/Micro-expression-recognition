import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import pickle

class BaseMicroExpressionDataset(Dataset):
    """微表情数据集基类，包含通用的处理逻辑"""
    def __init__(self, root_dir, num_frames=16, height=112, width=112, config=None, log_func=print):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.config = config
        self.log_func = log_func
        self.frame_step = config.frame_step if config else 1
        
        # 样本列表，由子类填充
        # 每个元素应为 {'video_path': str, 'label': int, 'video_name': str}
        self.samples = []
        
        # 类别名称，由子类定义
        self.class_names = []
        
        # 光流缓存目录（按数据集分类）
        dataset_name = os.path.basename(self.root_dir)
        self.flow_cache_dir = os.path.join('cache', 'optical_flow', dataset_name)
        os.makedirs(self.flow_cache_dir, exist_ok=True)
        


    def get_class_names(self):
        """获取数据集的类别名称"""
        return self.class_names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        label = sample['label']
        video_name = sample['video_name']
        
        # 加载并处理视频帧
        frames = self._load_and_process_frames(video_path, video_name)
        return frames, label

    def _apply_evm(self, frames, amplification=10.0, frequency_band=[0.1, 0.3], fps=30):
        """应用欧拉视频放大技术"""
        frames = np.array(frames, dtype=np.float32)
        n_frames = frames.shape[0]
        
        # 确保频率带有效
        try:
            # 确保frequency_band是一个可迭代对象
            if not hasattr(frequency_band, '__iter__') or len(frequency_band) < 2:
                raise ValueError("频率带参数必须是包含至少两个元素的可迭代对象")
            
            low, high = frequency_band
            # 确保low和high是有效的数值
            if low is None or high is None:
                raise ValueError("频率带参数无效")
            
            # 尝试转换为浮点数
            low = float(low)
            high = float(high)
            
            # 毫秒单位转换为Hz
            # 毫秒转Hz: Hz = 1000 / 毫秒
            low = 1000 / low
            high = 1000 / high
            # 确保low < high
            if low > high:
                low, high = high, low
            
            # 确保high > low > 0
            if low <= 0:
                low = 0.1
            if high <= 0:
                high = low + 0.1
            if high <= low:
                high = low + 0.1
        except Exception as e:
            # 如果频率带参数无效，使用默认值
            low = 0.1
            high = 0.3
            print(f"频率带参数无效，使用默认值：{low}, {high}")
        
        # 计算频率和创建带通滤波器（只计算一次）
        # 将实际频率（Hz）转换为归一化频率（0-0.5）
        max_freq = fps / 2  # 奈奎斯特频率
        low_norm = low / max_freq
        high_norm = high / max_freq
        
        # 确保归一化频率在有效范围内
        low_norm = max(0.01, min(low_norm, 0.49))
        high_norm = max(low_norm + 0.01, min(high_norm, 0.5))
        
        freq = np.fft.fftfreq(n_frames)
        mask = np.zeros_like(freq, dtype=np.float32)
        mask[(np.abs(freq) >= low_norm) & (np.abs(freq) <= high_norm)] = 1
        
        # 使用向量化操作处理整个视频
        # 傅里叶变换（沿时间轴）
        fft_frames = np.fft.fft(frames, axis=0)
        
        # 根据输入维度调整mask形状
        if len(frames.shape) == 4:
            # (n_frames, height, width, channels)
            mask = mask[:, np.newaxis, np.newaxis, np.newaxis]
        elif len(frames.shape) == 3:
            # (n_frames, height, width)
            mask = mask[:, np.newaxis, np.newaxis]
        
        # 应用带通滤波器
        filtered_fft = fft_frames * mask
        
        # 逆傅里叶变换得到微小变化
        small_changes = np.fft.ifft(filtered_fft, axis=0).real
        
        # 放大微小变化
        amplified_changes = small_changes * amplification
        
        # 将放大后的变化加回到原始信号
        result_frames = frames + amplified_changes
        
        return np.clip(result_frames, 0, 255).astype(np.uint8)
    


    def _calculate_optical_flow(self, prev_frame, curr_frame):
        """计算两帧之间的光流"""
        # 处理单通道图像
        if len(prev_frame.shape) == 2 or prev_frame.shape[2] == 1:
            prev_gray = prev_frame if len(prev_frame.shape) == 2 else prev_frame[:, :, 0]
        else:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        
        if len(curr_frame.shape) == 2 or curr_frame.shape[2] == 1:
            curr_gray = curr_frame if len(curr_frame.shape) == 2 else curr_frame[:, :, 0]
        else:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        
        if self.config and self.config.optical_flow_type == 'tv_l1':
            try:
                tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
                flow = tvl1.calc(prev_gray, curr_gray, None)
            except:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # 统一后处理：计算幅度并添加幅度通道
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # 取消归一化处理，保持原始光流值
        magnitude = magnitude[..., np.newaxis]
        return np.concatenate([flow, magnitude], axis=-1)

    def _apply_data_augmentation(self, frames, flow_frames=None):
        """应用数据增强"""
        if not self.config or not self.config.use_data_augmentation:
            return frames, flow_frames
        
        augmented_frames = []
        augmented_flow_frames = [] if flow_frames is not None else None
        # 随机参数对整个序列保持一致
        crop_params = None
        scale_params = None
        rotate_params = None
        flip = False
        brightness_params = None
        contrast_params = None
        
        try:
            if self.config.random_crop:
                h, w = self.height, self.width
                crop_size = self.config.crop_size
                if h - crop_size > 0:
                    top = np.random.randint(0, h - crop_size)
                    left = np.random.randint(0, w - crop_size)
                    crop_params = (top, left, crop_size)
                    
            if self.config.random_scale:
                scale_range = getattr(self.config, 'scale_range', [0.9, 1.1])
                if len(scale_range) == 2 and scale_range[0] > 0 and scale_range[1] > 0:
                    scale_params = np.random.uniform(*scale_range)
                else:
                    scale_params = 1.0
                    
            if self.config.random_rotation:
                rotation_range = getattr(self.config, 'rotation_range', [-3, 3])
                if len(rotation_range) == 2:
                    rotate_params = np.random.uniform(*rotation_range)
                else:
                    rotate_params = 0.0
                    
            if hasattr(self.config, 'random_horizontal_flip') and self.config.random_horizontal_flip:
                flip = np.random.random() > 0.5
            
            # 亮度增强
            if hasattr(self.config, 'random_brightness') and self.config.random_brightness:
                brightness_range = getattr(self.config, 'brightness_range', [0.8, 1.2])
                if len(brightness_range) == 2:
                    brightness_params = np.random.uniform(*brightness_range)
                else:
                    brightness_params = 1.0
            
            # 对比度增强
            if hasattr(self.config, 'random_contrast') and self.config.random_contrast:
                contrast_range = getattr(self.config, 'contrast_range', [0.8, 1.2])
                if len(contrast_range) == 2:
                    contrast_params = np.random.uniform(*contrast_range)
                else:
                    contrast_params = 1.0
            


            # 处理灰度帧
            for frame in frames:
                # 1. 旋转（在原始空间）
                if rotate_params:
                    angle = rotate_params
                    # 获取原始帧的尺寸
                    h, w = frame.shape[:2]
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                    frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
                
                # 2. 缩放（在原始空间）
                if scale_params:
                    scale = scale_params
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    if new_h <= 0 or new_w <= 0:
                        new_h, new_w = h, w
                    frame = cv2.resize(frame, (new_w, new_h))
                
                # 3. 翻转（在调整尺寸之前）
                if flip:
                    frame = cv2.flip(frame, 1)
                
                # 5. 裁剪/填充（保留原始空间关系）
                if crop_params:
                    t, l, s = crop_params
                    h, w = frame.shape[:2]
                    # 确保裁剪区域在有效范围内
                    t = min(t, h - s)
                    l = min(l, w - s)
                    t = max(0, t)
                    l = max(0, l)
                    
                    if len(frame.shape) == 3:
                        frame = frame[t:t+s, l:l+s, :]
                    else:
                        frame = frame[t:t+s, l:l+s]
                
                # 6. 调整到目标尺寸（最后调整尺寸）
                frame = cv2.resize(frame, (self.width, self.height))
                
                # 6. 颜色增强（合并类型转换）
                if brightness_params or contrast_params:
                    # 只进行一次类型转换
                    frame = frame.astype(np.float32)
                    
                    # 应用对比度增强（先调整对比度）
                    if contrast_params:
                        # 归一化到0-1范围
                        frame_norm = frame / 255.0
                        # 应用对比度调整
                        frame_norm = (frame_norm - 0.5) * contrast_params + 0.5
                        # 转换回0-255范围
                        frame = frame_norm * 255.0
                    
                    # 应用亮度增强（再调整亮度）
                    if brightness_params:
                        frame = frame * brightness_params
                    
                    # 裁剪到0-255范围并转换回uint8
                    frame = np.clip(frame, 0, 255)
                    frame = frame.astype(np.uint8)
                
                augmented_frames.append(frame)
            
            # 处理光流帧
            if flow_frames is not None:
                for flow in flow_frames:
                    # 1. 旋转（在原始空间）
                    if rotate_params:
                        angle = rotate_params
                        # 获取原始光流的尺寸
                        h, w = flow.shape[:2]
                        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                        # 对光流的每个通道单独进行旋转
                        flow_rotated = np.zeros_like(flow)
                        for c in range(flow.shape[2]):
                            flow_rotated[..., c] = cv2.warpAffine(flow[..., c], M, (w, h), flags=cv2.INTER_LINEAR)
                        flow = flow_rotated
                    
                    # 2. 缩放（在原始空间）
                    if scale_params:
                        scale = scale_params
                        h, w = flow.shape[:2]
                        new_h, new_w = int(h * scale), int(w * scale)
                        if new_h <= 0 or new_w <= 0:
                            new_h, new_w = h, w
                        flow = cv2.resize(flow, (new_w, new_h))
                        # 缩放光流值以保持物理意义
                        flow[..., :2] *= scale  # 只缩放x和y分量
                    
                    # 3. 翻转（在调整尺寸之前）
                    if flip:
                        flow = cv2.flip(flow, 1)
                        # 水平翻转时光流的x分量需要取反
                        flow[..., 0] *= -1
                    
                    # 4. 裁剪/填充（保留原始空间关系）
                    if crop_params:
                        t, l, s = crop_params
                        h, w = flow.shape[:2]
                        # 确保裁剪区域在有效范围内
                        t = min(t, h - s)
                        l = min(l, w - s)
                        t = max(0, t)
                        l = max(0, l)
                        flow = flow[t:t+s, l:l+s, :]
                    
                    # 5. 调整到目标尺寸（最后调整尺寸）
                    flow = cv2.resize(flow, (self.width, self.height))
                    # 调整光流值以保持物理意义
                    # 注意：这里不调整光流值，因为我们希望保持相对运动信息
                    
                    augmented_flow_frames.append(flow)
            
            if flow_frames is not None:
                return np.array(augmented_frames), np.array(augmented_flow_frames)
            else:
                return np.array(augmented_frames), False
        except Exception as e:
            if flow_frames is not None:
                return frames, flow_frames
            else:
                return frames, False

    def _load_and_process_frames(self, video_path, video_name):
        """通用的加载和处理流程，子类需提供关键帧索引逻辑"""
        # 生成缓存文件名，包含EVM参数
        if self.config:
            video_magnification = getattr(self.config, 'video_magnification', False)
            evm_amplification = getattr(self.config, 'evm_amplification', 10.0)
            evm_frequency_band = getattr(self.config, 'evm_frequency_band', [0.1, 0.3])
            fps = getattr(self.config, 'fps', 30)
            cache_filename = f"{video_name}_{self.num_frames}_{self.height}_{self.width}_{self.frame_step}_{self.config.optical_flow_type}_{self.config.use_two_stream}_{video_magnification}_{evm_amplification}_{evm_frequency_band[0]}_{evm_frequency_band[1]}_{fps}.pkl"
        else:
            cache_filename = f"{video_name}_{self.num_frames}_{self.height}_{self.width}_{self.frame_step}_none_false.pkl"
        cache_path = os.path.join(self.flow_cache_dir, cache_filename)
        
        # 加载帧文件列表
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
        if not frame_files:
            return torch.zeros((1, self.num_frames, self.height, self.width))
        
        # 检查缓存是否存在
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                augmented_gray_frames = cached_data['original_frames']
                flow_frames = cached_data['flow_frames']
                
                # 检查flow_frames的形状是否正确
                if flow_frames is not None and len(flow_frames) > 0:
                    # 检查是否为numpy数组
                    if isinstance(flow_frames, np.ndarray):
                        first_flow_shape = flow_frames[0].shape
                    else:
                        # 列表形式
                        first_flow_shape = flow_frames[0].shape
                    
                    if len(first_flow_shape) != 3 or first_flow_shape[2] != 3:
                        # 形状不正确，重新计算
                        self.log_func(f"缓存中的光流形状不正确: {first_flow_shape}，重新计算")
                        augmented_gray_frames, flow_frames = self._compute_frames(video_path, video_name, frame_files)
                    else:
                        # self.log_func(f"加载光流缓存: {cache_filename}")
                        pass
                else:
                    # 缓存为空，重新计算
                    augmented_gray_frames, flow_frames = self._compute_frames(video_path, video_name, frame_files)
            except Exception as e:
                self.log_func(f"缓存加载失败: {e}")
                # 缓存加载失败，重新计算
                augmented_gray_frames, flow_frames = self._compute_frames(video_path, video_name, frame_files)
        else:
            # 缓存不存在，计算光流
            augmented_gray_frames, flow_frames = self._compute_frames(video_path, video_name, frame_files)
            
            # 保存到缓存
            try:
                cached_data = {
                    'original_frames': augmented_gray_frames,
                    'flow_frames': flow_frames
                }
                with open(cache_path, 'wb') as f:
                    pickle.dump(cached_data, f)
                # self.log_func(f"保存光流缓存: {cache_filename}")
            except Exception as e:
                self.log_func(f"缓存保存失败: {e}")

        # 6. 归一化处理
        # 对灰度帧进行归一化
        normalized_gray_frames = augmented_gray_frames[:self.num_frames] / 255.0 * 2 - 1
        # 确保有通道维度
        if len(normalized_gray_frames.shape) == 3:
            normalized_gray_frames = np.expand_dims(normalized_gray_frames, axis=-1)
        
        # 对光流进行归一化（已经在_calculate_optical_flow中处理）
        # 将列表转换为numpy数组后再转换为张量，提高性能
        flow_frames_np = np.array(flow_frames)
        flow_frames_tensor = torch.tensor(flow_frames_np).permute(3, 0, 1, 2).float()

        if self.config and self.config.use_two_stream:
            # 7. 合并通道
            gray_tensor = torch.tensor(normalized_gray_frames).permute(3, 0, 1, 2).float()
            return torch.cat([gray_tensor, flow_frames_tensor], dim=0)
        
        # 单流法使用灰度图
        gray_tensor = torch.tensor(normalized_gray_frames).permute(3, 0, 1, 2).float()
        return gray_tensor
    
    def _compute_frames(self, video_path, video_name, frame_files):
        """计算帧和光流"""
        # 获取采样窗口（子类需实现或提供默认逻辑）
        start_idx = self._get_sampling_start_idx(video_name, frame_files)
        needed_frames_count = self.num_frames * self.frame_step + 1
        end_idx = min(len(frame_files), start_idx + needed_frames_count)
        
        selected_frames = frame_files[start_idx:end_idx]
        while len(selected_frames) < needed_frames_count:
            selected_frames.append(selected_frames[-1] if selected_frames else frame_files[0])

        # 1. 加载原始视频帧
        original_frames = []
        for i in range(0, len(selected_frames), self.frame_step):
            if len(original_frames) >= self.num_frames + 1: break
            frame = Image.open(os.path.join(video_path, selected_frames[i])).resize((self.width, self.height))
            frame_np = np.array(frame)
            original_frames.append(frame_np)
        
        while len(original_frames) < self.num_frames + 1:
            original_frames.append(original_frames[-1])

        # 2. 灰度转换
        gray_frames = []
        for frame in original_frames:
            if len(frame.shape) == 3:
                # 彩色图像转换为灰度
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                # 已经是灰度图
                gray_frame = frame
            gray_frames.append(gray_frame)
        
        # 转换为numpy数组并添加通道维度
        gray_frames = np.array(gray_frames)
        if len(gray_frames.shape) == 3:
            gray_frames = np.expand_dims(gray_frames, axis=-1)

        # 保存原始灰度帧用于计算光流
        original_gray_frames = gray_frames.copy()

        # 4. 计算 optical flow（使用原始灰度帧，在EVM和增强前）
        flow_frames = []
        for i in range(len(original_gray_frames) - 1):
            flow_frames.append(self._calculate_optical_flow(original_gray_frames[i], original_gray_frames[i+1]))

        # 3. 视频放大（EVM）
        if self.config:
            if hasattr(self.config, 'video_magnification') and self.config.video_magnification:
                amplification = getattr(self.config, 'evm_amplification', 10.0)
                frequency_band = getattr(self.config, 'evm_frequency_band', [0.1, 0.3])
                fps = getattr(self.config, 'fps', 30)  # 获取帧率，默认30fps
                
                # 在数据增强之前应用EVM，避免增强改变运动信息
                gray_frames = self._apply_evm(gray_frames, amplification, frequency_band, fps)
        
        while len(flow_frames) < self.num_frames:
            flow_frames.append(flow_frames[-1])



        # 6. 数据增强（同步作用在 gray + flow）
        augmented_gray_frames, augmented_flow_frames = self._apply_data_augmentation(gray_frames, flow_frames)
            
        return augmented_gray_frames, augmented_flow_frames

    def _get_sampling_start_idx(self, video_name, frame_files):
        """获取采样起始索引，默认从中间开始，子类可重写"""
        return max(0, len(frame_files) // 2 - (self.num_frames * self.frame_step) // 2)