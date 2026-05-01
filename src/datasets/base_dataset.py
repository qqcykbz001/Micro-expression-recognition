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
        
        # 光流缓存目录
        self.flow_cache_dir = os.path.join('cache')
        os.makedirs(self.flow_cache_dir, exist_ok=True)

        # 帧文件列表缓存
        self._frame_files_cache = {}
        


    def get_class_names(self):
        """获取数据集的类别名称"""
        return self.class_names

    def __len__(self):
        return len(self.samples)

    def _get_frame_files(self, video_path):
        """获取视频帧文件列表（首次扫描后缓存，避免每样本 os.listdir）"""
        if video_path not in self._frame_files_cache:
            try:
                with os.scandir(video_path) as entries:
                    self._frame_files_cache[video_path] = sorted(
                        e.name for e in entries
                        if e.name.endswith(('.jpg', '.png'))
                    )
            except OSError:
                return []
        return self._frame_files_cache[video_path]

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
            # 1. 先确定缩放（后续crop需基于缩放后尺寸计算）
            if self.config.random_scale:
                scale_range = getattr(self.config, 'scale_range', [0.9, 1.1])
                if len(scale_range) == 2 and scale_range[0] > 0 and scale_range[1] > 0:
                    scale_params = np.random.uniform(*scale_range)
                else:
                    scale_params = 1.0

            # 2. 基于缩放后尺寸计算crop（避免offset在缩放后偏位或跳过裁剪）
            if self.config.random_crop:
                scaled_h = int(self.height * (scale_params if scale_params else 1.0))
                scaled_w = int(self.width * (scale_params if scale_params else 1.0))
                crop_size = self.config.crop_size
                if scaled_h > crop_size and scaled_w > crop_size:
                    top = np.random.randint(0, scaled_h - crop_size)
                    left = np.random.randint(0, scaled_w - crop_size)
                    crop_params = (top, left, crop_size)

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
                # 1. 缩放（在原始空间，先缩放后旋转以减少旋转空角扩散）
                if scale_params:
                    scale = scale_params
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    if new_h <= 0 or new_w <= 0:
                        new_h, new_w = h, w
                    frame = cv2.resize(frame, (new_w, new_h))

                # 2. 旋转（在缩放后空间）
                if rotate_params:
                    angle = rotate_params
                    h, w = frame.shape[:2]
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                    frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)

                # 3. 翻转
                if flip:
                    frame = cv2.flip(frame, 1)

                # 4. 裁剪（在缩放+旋转后，可干净去除旋转空角）
                if crop_params:
                    t, l, s = crop_params
                    h, w = frame.shape[:2]
                    t = min(t, h - s)
                    l = min(l, w - s)
                    t = max(0, t)
                    l = max(0, l)

                    if len(frame.shape) == 3:
                        frame = frame[t:t+s, l:l+s, :]
                    else:
                        frame = frame[t:t+s, l:l+s]

                # 5. 调整到目标尺寸
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
                    # 1. 缩放（在原始空间，先缩放后旋转以减少旋转空角扩散）
                    if scale_params:
                        scale = scale_params
                        h, w = flow.shape[:2]
                        new_h, new_w = int(h * scale), int(w * scale)
                        if new_h <= 0 or new_w <= 0:
                            new_h, new_w = h, w
                        flow = cv2.resize(flow, (new_w, new_h))
                        flow[..., :2] *= scale

                    # 2. 旋转（在缩放后空间）
                    if rotate_params:
                        angle = rotate_params
                        h, w = flow.shape[:2]
                        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                        flow_rotated = np.zeros_like(flow)
                        for c in range(flow.shape[2]):
                            flow_rotated[..., c] = cv2.warpAffine(flow[..., c], M, (w, h), flags=cv2.INTER_LINEAR)
                        flow = flow_rotated
                        # 向量方向旋转补偿：dx/dy需随图像旋转
                        angle_rad = np.deg2rad(angle)
                        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                        dx = flow[..., 0].copy()
                        dy = flow[..., 1].copy()
                        flow[..., 0] = dx * cos_a - dy * sin_a
                        flow[..., 1] = dx * sin_a + dy * cos_a

                    # 3. 翻转
                    if flip:
                        flow = cv2.flip(flow, 1)
                        flow[..., 0] *= -1

                    # 4. 裁剪（在缩放+旋转后，可干净去除旋转空角）
                    if crop_params:
                        t, l, s = crop_params
                        h, w = flow.shape[:2]
                        t = min(t, h - s)
                        l = min(l, w - s)
                        t = max(0, t)
                        l = max(0, l)
                        flow = flow[t:t+s, l:l+s, :]

                    # 5. 调整到目标尺寸
                    resize_scale = self.width / flow.shape[1] if flow.shape[1] > 0 else 1.0
                    flow = cv2.resize(flow, (self.width, self.height))
                    flow[..., :2] *= resize_scale

                    augmented_flow_frames.append(flow)
            
            if flow_frames is not None:
                return np.array(augmented_frames), np.array(augmented_flow_frames)
            else:
                return np.array(augmented_frames), None
        except Exception as e:
            if flow_frames is not None:
                return frames, flow_frames
            else:
                return frames, None

    def _apply_temporal_augmentation(self, gray_frames, flow_frames):
        """时域增强：随机帧丢弃 + 时序局部打乱（灰度与光流的操作帧范围对齐）"""
        if not self.config:
            return gray_frames, flow_frames

        # 操作范围取 num_frames（光流帧数），避免灰度多出的一帧导致越界
        T = min(len(gray_frames), self.num_frames)
        if flow_frames is not None:
            T = min(T, len(flow_frames))
        if T < 4:
            return gray_frames, flow_frames

        # --- 随机帧丢弃：选一帧用相邻帧线性插值填充 ---
        if (getattr(self.config, 'random_frame_dropout', False)
                and np.random.random() < getattr(self.config, 'frame_dropout_prob', 0.3)):
            drop_idx = np.random.randint(1, T - 1)
            gray_frames[drop_idx] = ((gray_frames[drop_idx - 1].astype(np.float32)
                                      + gray_frames[drop_idx + 1].astype(np.float32)) / 2.0).astype(np.uint8)
            if flow_frames is not None and drop_idx < len(flow_frames) - 1:
                # 光流帧：对受影响的相邻光流做平滑
                flow_arr = flow_frames if isinstance(flow_frames, np.ndarray) else np.array(flow_frames)
                flow_arr[drop_idx] = ((flow_arr[drop_idx - 1].astype(np.float32)
                                       + flow_arr[min(drop_idx + 1, len(flow_arr) - 1)].astype(np.float32)) / 2.0)
                if not isinstance(flow_frames, np.ndarray):
                    for i in range(len(flow_frames)):
                        flow_frames[i] = flow_arr[i]

        # --- 时序局部打乱：2~3帧连续块反转顺序 ---
        if (getattr(self.config, 'random_temporal_shuffle', False)
                and np.random.random() < getattr(self.config, 'temporal_shuffle_prob', 0.3)):
            block_size = np.random.randint(2, 4)
            start = np.random.randint(0, T - block_size)
            idx_fwd = list(range(start, start + block_size))
            idx_rev = idx_fwd[::-1]
            gray_frames[idx_fwd] = gray_frames[idx_rev]
            if flow_frames is not None:
                if isinstance(flow_frames, np.ndarray):
                    flow_frames[idx_fwd] = flow_frames[idx_rev]
                else:
                    for i, j in zip(idx_fwd, idx_rev):
                        flow_frames[i], flow_frames[j] = flow_frames[j].copy(), flow_frames[i].copy()

        return gray_frames, flow_frames

    # 缓存版本号，格式变更时递增使旧缓存自动失效
    _CACHE_VERSION = 3

    def _load_and_process_frames(self, video_path, video_name):
        """通用的加载和处理流程，子类需提供关键帧索引逻辑"""
        # 生成缓存键（含版本号，旧版缓存自动失效）
        if self.config:
            video_magnification = getattr(self.config, 'video_magnification', False)
            evm_amplification = getattr(self.config, 'evm_amplification', 10.0)
            evm_frequency_band = getattr(self.config, 'evm_frequency_band', [0.1, 0.3])
            fps = getattr(self.config, 'fps', 30)
            use_two_stream = getattr(self.config, 'use_two_stream', False)
            cache_filename = (
                f"v{self._CACHE_VERSION}_{video_name}_n{self.num_frames}_"
                f"h{self.height}w{self.width}_s{self.frame_step}_"
                f"{use_two_stream}_{self.config.optical_flow_type}_"
                f"{video_magnification}_{evm_amplification}_"
                f"{evm_frequency_band[0]}_{evm_frequency_band[1]}_{fps}.pkl"
                
            )
        else:
            cache_filename = f"v{self._CACHE_VERSION}_{video_name}_{self.num_frames}_{self.height}_{self.width}_{self.frame_step}_none.pkl"
        cache_path = os.path.join(self.flow_cache_dir, cache_filename)

        # 加载帧文件列表（首次 scandir，后续命中缓存）
        frame_files = self._get_frame_files(video_path)
        if not frame_files:
            return torch.zeros((1, self.num_frames, self.height, self.width))

        need_flow = self.config and self.config.use_two_stream

        # 加载或计算 — 缓存只存原始数据，不含增强
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                raw_gray_frames = cached_data['raw_gray_frames']
                raw_flow_frames = cached_data['raw_flow_frames']

                # 形状校验（仅在需要光流时）
                if need_flow:
                    if raw_flow_frames is None or len(raw_flow_frames) == 0:
                        raw_gray_frames, raw_flow_frames = self._compute_raw_frames(
                            video_path, video_name, frame_files)
                    else:
                        raw_flow_frames = list(raw_flow_frames)
                        first_shape = raw_flow_frames[0].shape
                        if len(first_shape) != 3 or first_shape[2] != 3:
                            self.log_func(f"缓存光流形状异常 {first_shape}，重新计算")
                            raw_gray_frames, raw_flow_frames = self._compute_raw_frames(
                                video_path, video_name, frame_files)
            except Exception as e:
                self.log_func(f"缓存加载失败: {e}")
                raw_gray_frames, raw_flow_frames = self._compute_raw_frames(
                    video_path, video_name, frame_files)
        else:
            raw_gray_frames, raw_flow_frames = self._compute_raw_frames(
                video_path, video_name, frame_files)

            # 保存原始数据到缓存（不含增强）
            try:
                cached_data = {
                    'raw_gray_frames': raw_gray_frames,
                    'raw_flow_frames': raw_flow_frames,
                }
                with open(cache_path, 'wb') as f:
                    pickle.dump(cached_data, f)
            except Exception as e:
                self.log_func(f"缓存保存失败: {e}")

        # --- 以下每次 __getitem__ 都实时执行 ---

        # 数据增强（每次随机，不受缓存影响）
        raw_gray_frames, raw_flow_frames = self._apply_data_augmentation(
            raw_gray_frames, raw_flow_frames if need_flow else None)

        # 时域增强（空间增强后、归一化前，灰度与光流同步）
        if self.config and self.config.use_data_augmentation:
            raw_gray_frames, raw_flow_frames = self._apply_temporal_augmentation(
                raw_gray_frames, raw_flow_frames)

        # 灰度帧归一化
        normalized_gray = raw_gray_frames[:self.num_frames].astype(np.float32)
        normalized_gray /= np.float32(127.5)
        normalized_gray -= np.float32(1.0)
        if len(normalized_gray.shape) == 3:
            normalized_gray = np.expand_dims(normalized_gray, axis=-1)

        if need_flow:
            # 光流归一化：per-sample逐通道标准化
            flow_np = np.array(raw_flow_frames)
            flow_mean = flow_np.mean(axis=(0, 1, 2), keepdims=True)
            flow_std = flow_np.std(axis=(0, 1, 2), keepdims=True) + 1e-8
            flow_np = (flow_np - flow_mean) / flow_std
            flow_tensor = torch.tensor(flow_np).permute(3, 0, 1, 2).float()
            gray_tensor = torch.tensor(normalized_gray).permute(3, 0, 1, 2).float()
            return torch.cat([gray_tensor, flow_tensor], dim=0)

        gray_tensor = torch.tensor(normalized_gray).permute(3, 0, 1, 2).float()
        return gray_tensor
    
    def _compute_raw_frames(self, video_path, video_name, frame_files):
        """计算原始帧和光流（不含数据增强，增强在加载后实时应用）"""
        start_idx = self._get_sampling_start_idx(video_name, frame_files)
        needed_frames_count = self.num_frames * self.frame_step + 1
        end_idx = min(len(frame_files), start_idx + needed_frames_count)

        selected_frames = frame_files[start_idx:end_idx]
        while len(selected_frames) < needed_frames_count:
            selected_frames.append(selected_frames[-1] if selected_frames else frame_files[0])

        # 1. 加载原始视频帧
        original_frames = []
        for i in range(0, len(selected_frames), self.frame_step):
            if len(original_frames) >= self.num_frames + 1:
                break
            frame = Image.open(os.path.join(video_path, selected_frames[i])).resize(
                (self.width, self.height))
            original_frames.append(np.array(frame))

        while len(original_frames) < self.num_frames + 1:
            original_frames.append(original_frames[-1])

        # 2. 灰度转换
        gray_frames = []
        for frame in original_frames:
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray_frame = frame
            gray_frames.append(gray_frame)

        gray_frames = np.array(gray_frames)
        if len(gray_frames.shape) == 3:
            gray_frames = np.expand_dims(gray_frames, axis=-1)

        original_gray_frames = gray_frames.copy()

        # 3. 计算光流（基于原始灰度帧，不受EVM影响）
        need_flow = self.config and self.config.use_two_stream
        flow_frames = []
        if need_flow:
            for i in range(len(original_gray_frames) - 1):
                flow_frames.append(
                    self._calculate_optical_flow(original_gray_frames[i],
                                                 original_gray_frames[i + 1]))
            if len(flow_frames) == 0:
                flow_frames = [np.zeros((self.height, self.width, 3), dtype=np.float32)
                               for _ in range(self.num_frames)]
            while len(flow_frames) < self.num_frames:
                flow_frames.append(flow_frames[-1])

        # 4. EVM 视频放大（作用于灰度帧，光流基于未放大的帧）
        if self.config:
            if (hasattr(self.config, 'video_magnification')
                    and self.config.video_magnification):
                amplification = getattr(self.config, 'evm_amplification', 10.0)
                frequency_band = getattr(self.config, 'evm_frequency_band', [0.1, 0.3])
                fps = getattr(self.config, 'fps', 30)
                gray_frames = self._apply_evm(
                    gray_frames, amplification, frequency_band, fps)

        # 不在此处做增强 — 返回原始数据，增强由调用方实时应用
        return gray_frames, flow_frames

    def _get_sampling_start_idx(self, video_name, frame_files):
        """获取采样起始索引，默认从中间开始，子类可重写"""
        return max(0, len(frame_files) // 2 - (self.num_frames * self.frame_step) // 2)