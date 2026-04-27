#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化光流和视频放大效果

此脚本用于展示：
1. 原始视频帧 vs 灰度帧
2. 不同光流算法的效果
3. 视频放大技术的效果对比
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.configs.config import Config
from src.datasets.base_dataset import BaseMicroExpressionDataset

class VisualizationDataset(BaseMicroExpressionDataset):
    """用于可视化的数据集类"""
    def __init__(self, config):
        super().__init__(root_dir='', config=config)
    
    def _get_sampling_start_idx(self, video_name, frame_files):
        """获取采样起始索引"""
        return 0

def visualize_optical_flow(flow, title="Optical Flow"):
    """可视化光流"""
    # 计算光流幅度
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    # 计算光流方向
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    # 转换为HSV颜色空间
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (angle * 180 / np.pi / 2) % 180
    hsv[..., 1] = 255
    # 对幅度进行归一化，确保可视化效果
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # 转换回BGR颜色空间
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr

def visualize_flow_component(flow, component_idx, title):
    """可视化光流的单个分量（水平或垂直）"""
    # 提取指定分量
    component = flow[..., component_idx]
    # 对分量进行归一化，确保可视化效果
    normalized = cv2.normalize(component, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 转换为彩色图像（使用灰度映射）
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return colored

def save_frame(output_dir, filename, frame):
    """保存帧到文件"""
    # 直接处理原始帧，无论是RGB还是灰度图
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # RGB图像，转换为BGR格式保存
        cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    else:
        # 灰度图像，直接保存
        if len(frame.shape) == 3:
            frame = frame[..., 0]
        cv2.imwrite(os.path.join(output_dir, filename), frame)

def create_frame_sequence(output_dir, frames, name):
    """创建帧序列的可视化"""
    # 创建序列目录
    sequence_dir = os.path.join(output_dir, f'{name}_sequence')
    os.makedirs(sequence_dir, exist_ok=True)
    
    # 保存每帧
    for i, frame in enumerate(frames):
        save_frame(sequence_dir, f'{i:03d}.jpg', frame)
    
    # 创建视频
    if frames:
        # 确定视频参数
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(os.path.join(output_dir, f'{name}_sequence.avi'), fourcc, 5, (width, height))
        
        for frame in frames:
            # 灰度图转BGR
            if len(frame.shape) == 2:
                out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                # RGB转BGR
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                # 其他情况，尝试从第一通道转换
                out.write(cv2.cvtColor(frame[..., 0], cv2.COLOR_GRAY2BGR))
        
        out.release()

def create_comparison_video(output_dir, original_frames, gray_frames, flow_frames, magnified_frames):
    """创建对比视频，将不同处理阶段的帧合并在一起"""
    if not original_frames:
        return
    
    # 确定视频参数
    height, width = original_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(os.path.join(output_dir, 'comparison_video.avi'), fourcc, 5, (width * 2, height * 2))
    
    # 确保所有帧序列长度一致
    min_length = len(original_frames)
    
    # 检查gray_frames是否有效
    if gray_frames is not None:
        if hasattr(gray_frames, '__len__'):
            min_length = min(min_length, len(gray_frames))
    
    # 检查flow_frames是否有效
    if flow_frames is not None:
        if hasattr(flow_frames, '__len__'):
            min_length = min(min_length, len(flow_frames))
    
    # 检查magnified_frames是否有效
    if magnified_frames is not None and hasattr(magnified_frames, '__len__'):
        min_length = min(min_length, len(magnified_frames))
    
    # 处理每帧
    for i in range(min_length):
        # 原始帧
        original = original_frames[i]
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        # 视频放大帧
        magnified = np.zeros_like(original)
        if magnified_frames is not None and hasattr(magnified_frames, '__len__') and i < len(magnified_frames):
            mag_frame = magnified_frames[i]
            if len(mag_frame.shape) == 3:
                mag_frame = mag_frame[..., 0]
            # 归一化到0-255
            mag_frame = cv2.normalize(mag_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            magnified = cv2.cvtColor(mag_frame, cv2.COLOR_GRAY2BGR)
        
        # 光流x方向图
        flow_x_vis = np.zeros_like(original)
        if flow_frames is not None and hasattr(flow_frames, '__len__') and i < len(flow_frames):
            flow_x_vis = visualize_flow_component(flow_frames[i], 0, 'Horizontal Flow')
        
        # 光流y方向图
        flow_y_vis = np.zeros_like(original)
        if flow_frames is not None and hasattr(flow_frames, '__len__') and i < len(flow_frames):
            flow_y_vis = visualize_flow_component(flow_frames[i], 1, 'Vertical Flow')
        
        # 调整所有帧的大小一致
        original = cv2.resize(original, (width, height))
        magnified = cv2.resize(magnified, (width, height))
        flow_x_vis = cv2.resize(flow_x_vis, (width, height))
        flow_y_vis = cv2.resize(flow_y_vis, (width, height))
        
        # 创建对比帧（2x2网格）：左上原图，右上视频放大，左下光流x，右下光流y
        top_row = np.hstack((original, magnified))
        bottom_row = np.hstack((flow_x_vis, flow_y_vis))
        comparison_frame = np.vstack((top_row, bottom_row))
        
        # 写入视频
        out.write(comparison_frame)
    
    out.release()

def visualize_frames(original_frames, gray_frames, flow_frames, magnified_frames, video_name):
    """可视化不同处理阶段的帧"""
    # 创建输出目录
    output_dir = f'visualization_output/{video_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择中间帧进行可视化
    middle_idx = len(original_frames) // 2
    
    # 原始帧（灰度图）
    original_frame = original_frames[middle_idx]
    save_frame(output_dir, 'original_frame.jpg', original_frame)
    
    # 灰度帧（与原始帧相同，因为输入已经是灰度图）
    gray_frame = gray_frames[middle_idx]
    save_frame(output_dir, 'gray_frame.jpg', gray_frame)
    
    # 光流
    flow_vis = None
    horizontal_flow_vis = None
    vertical_flow_vis = None
    if flow_frames and len(flow_frames) > 0:
        flow_idx = min(middle_idx, len(flow_frames) - 1)
        flow_frame = flow_frames[flow_idx]
        # 可视化完整光流
        flow_vis = visualize_optical_flow(flow_frame)
        cv2.imwrite(os.path.join(output_dir, 'optical_flow.jpg'), flow_vis)
        # 可视化水平方向光流
        horizontal_flow_vis = visualize_flow_component(flow_frame, 0, 'Horizontal Flow')
        cv2.imwrite(os.path.join(output_dir, 'horizontal_flow.jpg'), horizontal_flow_vis)
        # 可视化垂直方向光流
        vertical_flow_vis = visualize_flow_component(flow_frame, 1, 'Vertical Flow')
        cv2.imwrite(os.path.join(output_dir, 'vertical_flow.jpg'), vertical_flow_vis)
    
    # 视频放大帧
    magnified_frame = None
    if magnified_frames is not None and hasattr(magnified_frames, '__len__') and len(magnified_frames) > 0:
        magnified_frame = magnified_frames[middle_idx]
        if len(magnified_frame.shape) == 3:
            magnified_frame = magnified_frame[..., 0]
        # 归一化到0-255
        magnified_frame = cv2.normalize(magnified_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'magnified_frame.jpg'), magnified_frame)
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 原始帧（灰度图）
    axes[0, 0].imshow(original_frame, cmap='gray')
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    # 视频放大帧
    if magnified_frame is not None:
        axes[0, 1].imshow(magnified_frame, cmap='gray')
        axes[0, 1].set_title('Magnified Frame')
        axes[0, 1].axis('off')
    else:
        axes[0, 1].set_visible(False)
    
    # 光流x方向图
    if horizontal_flow_vis is not None:
        axes[1, 0].imshow(cv2.cvtColor(horizontal_flow_vis, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Horizontal Flow (X)')
        axes[1, 0].axis('off')
    else:
        axes[1, 0].set_visible(False)
    
    # 光流y方向图
    if vertical_flow_vis is not None:
        axes[1, 1].imshow(cv2.cvtColor(vertical_flow_vis, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Vertical Flow (Y)')
        axes[1, 1].axis('off')
    else:
        axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.jpg'))
    plt.close()
    
    # 创建光流分量对比图
    if flow_vis is not None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 完整光流
        axes[0].imshow(cv2.cvtColor(flow_vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Full Optical Flow')
        axes[0].axis('off')
        
        # 水平方向光流
        if horizontal_flow_vis is not None:
            axes[1].imshow(cv2.cvtColor(horizontal_flow_vis, cv2.COLOR_BGR2RGB))
            axes[1].set_title('Horizontal Flow')
            axes[1].axis('off')
        else:
            axes[1].set_visible(False)
        
        # 垂直方向光流
        if vertical_flow_vis is not None:
            axes[2].imshow(cv2.cvtColor(vertical_flow_vis, cv2.COLOR_BGR2RGB))
            axes[2].set_title('Vertical Flow')
            axes[2].axis('off')
        else:
            axes[2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'flow_components.jpg'))
        plt.close()
    
    # 创建整个帧序列的可视化
    create_frame_sequence(output_dir, original_frames, 'original')
    
    # 创建灰度帧序列
    gray_frames_visual = []
    for frame in gray_frames:
        if len(frame.shape) == 3:
            gray_frames_visual.append(frame[..., 0])
        else:
            gray_frames_visual.append(frame)
    create_frame_sequence(output_dir, gray_frames_visual, 'gray')
    
    # 创建光流序列
    if flow_frames and len(flow_frames) > 0:
        flow_vis_frames = []
        for flow_frame in flow_frames:
            flow_vis = visualize_optical_flow(flow_frame)
            flow_vis_frames.append(cv2.cvtColor(flow_vis, cv2.COLOR_BGR2RGB))
        create_frame_sequence(output_dir, flow_vis_frames, 'optical_flow')
    
    # 创建视频放大序列
    if magnified_frames is not None and hasattr(magnified_frames, '__len__') and len(magnified_frames) > 0:
        magnified_frames_visual = []
        for frame in magnified_frames:
            if len(frame.shape) == 3:
                frame = frame[..., 0]
            # 归一化到0-255
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            magnified_frames_visual.append(frame)
        create_frame_sequence(output_dir, magnified_frames_visual, 'magnified')
    
    # 创建对比视频
    create_comparison_video(output_dir, original_frames, gray_frames, flow_frames, magnified_frames)
    
    print(f"可视化结果已保存到: {output_dir}")

def process_video(video_path, video_name, config):
    """处理单个视频并可视化效果"""
    print(f"处理视频: {video_name}")
    
    # 创建数据集实例
    dataset = VisualizationDataset(config)
    
    # 加载和处理帧
    try:
        # 计算帧和光流
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
        if not frame_files:
            print(f"视频 {video_name} 没有帧文件")
            return
        
        # 加载原始帧（灰度图）
        original_frames = []
        for frame_file in frame_files[:config.num_frames + 1]:
            frame = cv2.imread(os.path.join(video_path, frame_file), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                print(f"无法读取帧文件: {frame_file}")
                continue
            frame = cv2.resize(frame, (config.width, config.height))
            original_frames.append(frame)
        
        if not original_frames:
            print(f"视频 {video_name} 没有有效帧")
            return
        
        # 转换为灰度（直接使用灰度图）
        gray_frames = []
        for frame in original_frames:
            # 直接使用灰度图，添加通道维度
            gray = np.expand_dims(frame, axis=-1)
            gray_frames.append(gray)
        gray_frames = np.array(gray_frames)
        
        # 应用视频放大（在数据增强之前）
        magnified_frames = None
        if config.video_magnification:
            # 只使用EVM视频放大技术
            try:
                magnified_frames = dataset._apply_evm(gray_frames, config.evm_amplification, config.evm_frequency_band, config.fps)
            except Exception as e:
                print(f"应用视频放大时出错: {e}")
                magnified_frames = None
        
        # 确定用于计算光流的帧
        pre_augmentation_frames = gray_frames
        # pre_augmentation_frames = gray_frames if magnified_frames is None else magnified_frames
        
        # 计算光流（在数据增强之前）
        flow_frames = []
        for i in range(len(pre_augmentation_frames) - 1):
            try:
                flow = dataset._calculate_optical_flow(pre_augmentation_frames[i], pre_augmentation_frames[i+1])
                flow_frames.append(flow)
            except Exception as e:
                print(f"计算光流时出错: {e}")
                # 使用零矩阵作为替代
                flow = np.zeros((config.height, config.width, 3))
                flow_frames.append(flow)
        
        # 应用数据增强
        augmented_gray_frames, _ = dataset._apply_data_augmentation(pre_augmentation_frames)
        
        # 可视化效果
        visualize_frames(original_frames, augmented_gray_frames, flow_frames, magnified_frames, video_name)
        
    except Exception as e:
        print(f"处理视频 {video_name} 时出错: {e}")

def main():
    """主函数"""
    # 创建配置
    config = Config()
    
    # 获取数据集路径
    dataset_path = config.root_dir
    
    if not os.path.exists(dataset_path):
        print(f"数据集路径不存在: {dataset_path}")
        return
    
    # 处理视频
    processed_count = 0
    max_process = 3
    
    # 遍历子目录
    for root, dirs, files in os.walk(dataset_path):
        # 检查当前目录是否包含jpg或png文件
        frame_files = [f for f in files if f.endswith(('.jpg', '.png'))]
        if frame_files and processed_count < max_process:
            video_name = os.path.basename(root)
            process_video(root, video_name, config)
            processed_count += 1
            if processed_count >= max_process:
                break
    
    print("可视化完成！")

if __name__ == "__main__":
    main()