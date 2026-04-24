import os
import pandas as pd
import numpy as np
from collections import Counter
from src.datasets.base_dataset import BaseMicroExpressionDataset

class SAMMDataset(BaseMicroExpressionDataset):
    """SAMM 数据集实现"""
    def __init__(self, root_dir, num_frames=16, height=112, width=112, 
                 include_subjects=None, exclude_subjects=None, config=None, log_func=print):
        super(SAMMDataset, self).__init__(root_dir, num_frames, height, width, config, log_func)
        
        self.include_subjects = include_subjects
        self.exclude_subjects = exclude_subjects
        
        # 定义类别名称（三分类）
        self.class_names = ['Positive', 'Surprise', 'Negative']
        
        # 标签映射
        self.label_mapping = {
            'happiness': 0,
            'surprise': 1,
            'sadness': 2,
            'disgust': 2,
            'anger': 2,
            'fear': 2,
            'contempt': 2,
            'other': 2
        }
        
        # 获取标注文件路径
        self.excel_path = getattr(config, 'samm_excel_path', 
                                os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                '..', '..', 'SAMM', 'SAMM.xlsx'))
        
        # 加载标注
        self.annotations = self._load_annotations()
        
        # 收集样本
        self._collect_samples()
        
        # 统计分布
        self._log_distribution()

    def _load_annotations(self):
        """加载SAMM标注文件"""
        if not os.path.exists(self.excel_path):
            self.log_func(f"警告: 未找到标注文件 {self.excel_path}")
            return {}
            
        try:
            df = pd.read_excel(self.excel_path)
            annotations = {}
            
            # 处理不同的列名格式
            filename_col = 'Filename' if 'Filename' in df.columns else 'filename'
            emotion_col = 'Estimated Emotion' if 'Estimated Emotion' in df.columns else 'emotion'
            
            for _, row in df.iterrows():
                filename = row[filename_col]
                # 将情绪标签转换为小写以匹配映射
                emotion = row[emotion_col].lower() if isinstance(row[emotion_col], str) else row[emotion_col]
                annotations[filename] = {
                    'emotion': emotion,
                    'onset': row.get('OnsetFrame', row.get('onset', row.get('Onset'))),
                    'apex': row.get('ApexFrame', row.get('apex', row.get('Apex'))),
                    'offset': row.get('OffsetFrame', row.get('offset', row.get('Offset')))
                }
            self.log_func(f"成功加载标注文件，包含 {len(annotations)} 个样本")
            return annotations
        except Exception as e:
            self.log_func(f"加载标注文件失败: {e}")
            return {}

    def _collect_samples(self):
        """遍历目录收集有效样本"""
        total_videos = 0
        matched_videos = 0
        
        for subject in sorted(os.listdir(self.root_dir)):
            if self.include_subjects and subject not in self.include_subjects: continue
            if self.exclude_subjects and subject in self.exclude_subjects: continue
            
            subject_path = os.path.join(self.root_dir, subject)
            if not os.path.isdir(subject_path): continue
            
            for video in os.listdir(subject_path):
                video_path = os.path.join(subject_path, video)
                if not os.path.isdir(video_path): continue
                
                total_videos += 1
                
                # 检查视频是否在标注中
                if video in self.annotations:
                    emotion = self.annotations[video]['emotion']
                    # 映射情绪到三分类
                    if emotion in self.label_mapping:
                        self.samples.append({
                            'video_path': video_path,
                            'label': self.label_mapping[emotion],
                            'video_name': video
                        })
                        matched_videos += 1
                # 尝试不同的视频名称格式
                else:
                    # 尝试移除后缀或其他处理
                    matched = False
                    for key in self.annotations.keys():
                        if video in key or key in video:
                            emotion = self.annotations[key]['emotion']
                            if emotion in self.label_mapping:
                                self.samples.append({
                                    'video_path': video_path,
                                    'label': self.label_mapping[emotion],
                                    'video_name': video
                                })
                                matched_videos += 1
                                matched = True
                                break
                    if not matched:
                        # 尝试更灵活的匹配方式
                        for key in self.annotations.keys():
                            # 移除可能的后缀或前缀
                            video_clean = video.replace('_', '').replace('-', '')
                            key_clean = key.replace('_', '').replace('-', '')
                            if video_clean in key_clean or key_clean in video_clean:
                                emotion = self.annotations[key]['emotion']
                                if emotion in self.label_mapping:
                                    self.samples.append({
                                        'video_path': video_path,
                                        'label': self.label_mapping[emotion],
                                        'video_name': video
                                    })
                                    matched_videos += 1
                                    break
        
        self.log_func(f"视频目录总数: {total_videos}")
        self.log_func(f"成功匹配的视频: {matched_videos}")
        self.log_func(f"样本总数: {len(self.samples)}")

    def _get_sampling_start_idx(self, video_name, frame_files):
        """覆盖基类，以Apex帧为中心计算采样起始点"""
        apex_idx = len(frame_files) // 2  # 默认中心位置
        if video_name in self.annotations:
            apex_frame = self.annotations[video_name].get('ApexFrame')
            if apex_frame is not None:
                try:
                    apex_frame = int(apex_frame)
                    # 改进匹配逻辑：支持 006_05562.jpg 等格式
                    target_suffix = f"_{apex_frame:05d}"  # SAMM格式通常是5位数字
                    for i, f in enumerate(frame_files):
                        if target_suffix in f:
                            apex_idx = i
                            break
                except: pass
        
        # 计算以apex为中心的起始索引
        window_size = self.num_frames * self.frame_step
        start_idx = apex_idx - window_size // 2
        
        # 时域增强
        if self.config and self.config.use_data_augmentation:
            start_offset = np.random.randint(-2, 3)
            start_idx = max(0, start_idx + start_offset)
        else:
            start_idx = max(0, start_idx)
        
        return start_idx

    def _log_distribution(self):
        """记录类别分布并给出alpha建议"""
        if not self.samples: return
        labels = [s['label'] for s in self.samples]
        counts = Counter(labels)
        total = len(labels)
        self.log_func(f"\n数据集 {self.__class__.__name__} 类别分布:")
        for label in sorted(counts.keys()):
            self.log_func(f'  类别 {label}: {counts[label]} 个样本, 占比: {counts[label]/total:.4f}')
        
        # Alpha 建议
        weights = [total / counts[i] if i in counts else 1.0 for i in range(max(counts.keys()) + 1)]
        mean_w = sum(weights) / len(weights)
        self.log_func(f'  建议 focal_alpha: {[round(w/mean_w, 2) for w in weights]}')