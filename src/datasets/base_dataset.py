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

    def _apply_evm(self, frames, amplification=10.0, frequency_band=[0.1, 0.3]):
        """应用欧拉视频放大技术"""
        frames = np.array(frames, dtype=np.float32)
        fft_frames = np.fft.fft(frames, axis=0)
        n_frames = frames.shape[0]
        freq = np.fft.fftfreq(n_frames)
        mask = np.zeros_like(freq)
        low, high = frequency_band
        mask[(np.abs(freq) >= low) & (np.abs(freq) <= high)] = 1
        # 根据输入的维度调整mask形状
        if len(frames.shape) == 4:
            mask = mask[:, np.newaxis, np.newaxis, np.newaxis]
        elif len(frames.shape) == 3:
            mask = mask[:, np.newaxis, np.newaxis]
        fft_frames *= mask
        amplified_frames = np.fft.ifft(fft_frames, axis=0).real
        amplified_frames *= amplification
        result_frames = frames + amplified_frames
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
        
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        max_magnitude = np.max(magnitude)
        if max_magnitude > 0:
            flow = flow / max_magnitude
        
        magnitude = magnitude[..., np.newaxis]
        return np.concatenate([flow, magnitude], axis=-1)

    def _apply_data_augmentation(self, flow_frames):
        """应用数据增强"""
        if not self.config or not self.config.use_data_augmentation:
            return flow_frames
        
        augmented_frames = []
        # 随机参数对整个序列保持一致
        crop_params = None
        scale_params = None
        rotate_params = None
        
        if self.config.random_crop:
            h, w = self.height, self.width
            crop_size = self.config.crop_size
            top = np.random.randint(0, h - crop_size)
            left = np.random.randint(0, w - crop_size)
            crop_params = (top, left, crop_size)
            
        if self.config.random_scale:
            scale_params = np.random.uniform(*self.config.scale_range)
            
        if self.config.random_rotation:
            rotate_params = np.random.uniform(*self.config.rotation_range)

        for flow_frame in flow_frames:
            # 应用裁剪
            if crop_params:
                t, l, s = crop_params
                flow_frame = flow_frame[t:t+s, l:l+s, :]
                flow_frame = cv2.resize(flow_frame, (self.width, self.height))
            
            # 应用缩放
            if scale_params:
                scale = scale_params
                h, w = self.height, self.width
                new_h, new_w = int(h * scale), int(w * scale)
                flow_frame = cv2.resize(flow_frame, (new_w, new_h))
                if scale > 1:
                    top, left = (new_h - h) // 2, (new_w - w) // 2
                    flow_frame = flow_frame[top:top+h, left:left+w, :]
                else:
                    new_flow_frame = np.zeros((h, w, 3), dtype=flow_frame.dtype)
                    top, left = (h - new_h) // 2, (w - new_w) // 2
                    new_flow_frame[top:top+new_h, left:left+new_w, :] = flow_frame
                    flow_frame = new_flow_frame
            
            # 应用旋转
            if rotate_params:
                angle = rotate_params
                M = cv2.getRotationMatrix2D((self.width // 2, self.height // 2), angle, 1.0)
                flow_frame = cv2.warpAffine(flow_frame, M, (self.width, self.height), flags=cv2.INTER_LINEAR)
            
            augmented_frames.append(flow_frame)
        return np.array(augmented_frames)

    def _load_and_process_frames(self, video_path, video_name):
        """通用的加载和处理流程，子类需提供关键帧索引逻辑"""
        # 生成缓存文件名
        cache_filename = f"{video_name}_{self.num_frames}_{self.height}_{self.width}_{self.frame_step}_{self.config.optical_flow_type}_{self.config.use_two_stream}.pkl"
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
                original_frames = cached_data['original_frames']
                flow_frames = cached_data['flow_frames']
                # self.log_func(f"加载光流缓存: {cache_filename}")
            except Exception as e:
                self.log_func(f"缓存加载失败: {e}")
                # 缓存加载失败，重新计算
                original_frames, flow_frames = self._compute_frames(video_path, video_name, frame_files)
        else:
            # 缓存不存在，计算光流
            original_frames, flow_frames = self._compute_frames(video_path, video_name, frame_files)
            
            # 保存到缓存
            try:
                cached_data = {
                    'original_frames': original_frames,
                    'flow_frames': flow_frames
                }
                with open(cache_path, 'wb') as f:
                    pickle.dump(cached_data, f)
                # self.log_func(f"保存光流缓存: {cache_filename}")
            except Exception as e:
                self.log_func(f"缓存保存失败: {e}")

        flow_frames = self._apply_data_augmentation(flow_frames)
        flow_frames_tensor = torch.tensor(flow_frames).permute(3, 0, 1, 2).float()

        if self.config and self.config.use_two_stream:
            rgb_frames = original_frames[:self.num_frames] / 255.0 * 2 - 1
            rgb_tensor = torch.tensor(rgb_frames).permute(3, 0, 1, 2).float()
            return torch.cat([rgb_tensor, flow_frames_tensor], dim=0)
        
        # 单流法使用灰度图
        rgb_frames = original_frames[:self.num_frames] / 255.0 * 2 - 1
        rgb_tensor = torch.tensor(rgb_frames).permute(3, 0, 1, 2).float()
        return rgb_tensor
    
    def _compute_frames(self, video_path, video_name, frame_files):
        """计算帧和光流"""
        # 获取采样窗口（子类需实现或提供默认逻辑）
        start_idx = self._get_sampling_start_idx(video_name, frame_files)
        needed_frames_count = self.num_frames * self.frame_step + 1
        end_idx = min(len(frame_files), start_idx + needed_frames_count)
        
        selected_frames = frame_files[start_idx:end_idx]
        while len(selected_frames) < needed_frames_count:
            selected_frames.append(selected_frames[-1] if selected_frames else frame_files[0])

        original_frames = []
        for i in range(0, len(selected_frames), self.frame_step):
            if len(original_frames) >= self.num_frames + 1: break
            frame = Image.open(os.path.join(video_path, selected_frames[i])).resize((self.width, self.height))
            # 转换为灰度图
            frame = frame.convert('L')
            frame_np = np.array(frame)
            # 保持单通道
            original_frames.append(frame_np)
        
        while len(original_frames) < self.num_frames + 1:
            original_frames.append(original_frames[-1])

        # 转换为numpy数组并添加通道维度
        original_frames = np.array(original_frames)
        if len(original_frames.shape) == 3:
            original_frames = np.expand_dims(original_frames, axis=-1)

        if self.config and self.config.use_evm:
            original_frames = self._apply_evm(original_frames, self.config.evm_amplification, self.config.evm_frequency_band)

        flow_frames = []
        for i in range(len(original_frames) - 1):
            flow_frames.append(self._calculate_optical_flow(original_frames[i], original_frames[i+1]))
        
        while len(flow_frames) < self.num_frames:
            flow_frames.append(flow_frames[-1])
            
        return original_frames, flow_frames

    def _get_sampling_start_idx(self, video_name, frame_files):
        """获取采样起始索引，默认从中间开始，子类可重写"""
        return max(0, len(frame_files) // 2 - (self.num_frames * self.frame_step) // 2)