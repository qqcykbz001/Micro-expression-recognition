import os
import pandas as pd
import numpy as np
from collections import Counter
from src.datasets.base_dataset import BaseMicroExpressionDataset

class CASME2Dataset(BaseMicroExpressionDataset):
    """CASME II 数据集实现"""
    def __init__(self, root_dir, num_frames=16, height=112, width=112, 
                 include_subjects=None, exclude_subjects=None, config=None, log_func=print):
        super(CASME2Dataset, self).__init__(root_dir, num_frames, height, width, config, log_func)
        
        self.include_subjects = include_subjects
        self.exclude_subjects = exclude_subjects
        
        # 定义类别名称
        self.class_names = ['Positive', 'Surprise', 'Negative']
        
        # 标签映射
        self.label_mapping = {
            'happiness': 0,
            'surprise': 1,
            'sadness': 2,
            'disgust': 2,
            'anger': 2,
            'fear': 2
        }
        
        # 获取标注文件路径 (如果config未提供，则使用默认路径)
        self.excel_path = getattr(config, 'casme2_excel_path', 
                                os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                '..', '..', 'CASME2', 'CASME2.xlsx'))
        
        # 加载标注
        self.annotations = self._load_annotations()
        
        # 收集样本
        self._collect_samples()
        
        # 统计分布
        self._log_distribution()

    def _load_annotations(self):
        """加载CASME2.xlsx"""
        if not os.path.exists(self.excel_path):
            self.log_func(f"警告: 未找到标注文件 {self.excel_path}")
            return {}
            
        df = pd.read_excel(self.excel_path)
        annotations = {}
        for _, row in df.iterrows():
            filename = row['Filename']
            annotations[filename] = {
                'emotion': row['Estimated Emotion'],
                'onset': row.get('OnsetFrame', row.get('onset', row.get('Onset'))),
                'apex': row.get('ApexFrame', row.get('apex', row.get('Apex'))),
                'offset': row.get('OffsetFrame', row.get('offset', row.get('Offset')))
            }
        return annotations

    def _collect_samples(self):
        """遍历目录收集有效样本"""
        for subject in sorted(os.listdir(self.root_dir)):
            if self.include_subjects and subject not in self.include_subjects: continue
            if self.exclude_subjects and subject in self.exclude_subjects: continue
            
            subject_path = os.path.join(self.root_dir, subject)
            if not os.path.isdir(subject_path): continue
            
            for video in os.listdir(subject_path):
                video_path = os.path.join(subject_path, video)
                if not os.path.isdir(video_path): continue
                
                if video in self.annotations:
                    emotion = self.annotations[video]['emotion']
                    if emotion in self.label_mapping:
                        self.samples.append({
                            'video_path': video_path,
                            'label': self.label_mapping[emotion],
                            'video_name': video
                        })

    def _get_sampling_start_idx(self, video_name, frame_files):
        """覆盖基类，根据Onset/Apex计算采样起始点"""
        onset_idx = 0
        if video_name in self.annotations:
            onset_frame = self.annotations[video_name].get('onset')
            if onset_frame is not None:
                try:
                    onset_frame = int(onset_frame)
                    # 改进匹配逻辑：支持 img1.jpg, img001.jpg, img_001.jpg 等格式
                    target_suffix = f"{onset_frame:03d}"  # 常见的补零格式
                    for i, f in enumerate(frame_files):
                        # 检查文件名中是否包含该数字（作为独立部分）
                        if f'img{onset_frame}.' in f or f'_{onset_frame}.' in f or target_suffix in f:
                            onset_idx = i
                            break
                except: pass
        
        # 时域增强
        if self.config and self.config.use_data_augmentation:
            start_offset = np.random.randint(-2, 3)
            return max(0, onset_idx + start_offset)
        return onset_idx

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
