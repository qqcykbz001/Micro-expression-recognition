#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计数据集每个fold的类别分布

此脚本用于分析LOSO交叉验证中每个fold的训练集和测试集的类别分布情况，
帮助了解数据集的类别平衡性和可能对UAR/UF1计算的影响。
"""

import os
import numpy as np
from collections import Counter
from src.configs.config import Config
from src.datasets import get_dataset

# 日志写入函数
def log(message):
    """打印日志信息"""
    print(f"[INFO] {message}")

def analyze_fold_distribution():
    """分析每个fold的类别分布"""
    # 初始化配置
    config = Config()
    
    # 获取数据集路径
    root_dir = config.root_dir
    
    if not os.path.exists(root_dir):
        log(f"数据集路径不存在: {root_dir}")
        return
    
    # 获取所有受试者列表
    subjects = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    log(f'发现 {len(subjects)} 个受试者: {subjects}')
    
    # 类别名称映射
    class_names = {0: '类别0', 1: '类别1', 2: '类别2'}  # 根据实际类别名称修改
    
    # 存储每个fold的类别分布
    fold_distributions = []
    
    # 执行LOSO交叉验证分析
    for i, test_subject in enumerate(subjects):
        log(f'分析 Fold {i + 1}/{len(subjects)}, 测试受试者: {test_subject}')
        
        # 创建训练集和测试集
        exclude_subjects = [test_subject]
        train_dataset = get_dataset(config, exclude_subjects=exclude_subjects, log_func=log)
        
        # 创建测试集配置，禁用数据增强
        test_config = Config()
        # 复制所有配置参数
        for attr in dir(config):
            if not attr.startswith('__') and not callable(getattr(config, attr)):
                setattr(test_config, attr, getattr(config, attr))
        # 禁用数据增强
        test_config.use_data_augmentation = False
        test_dataset = get_dataset(test_config, include_subjects=exclude_subjects, log_func=log)
        
        log(f'训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}')
        
        # 检查测试集是否为空
        if len(test_dataset) == 0:
            log(f'警告: 测试集为空，跳过该fold')
            continue
        
        # 统计训练集类别分布
        train_labels = [sample['label'] for sample in train_dataset.samples]
        train_counter = Counter(train_labels)
        
        # 统计测试集类别分布
        test_labels = [sample['label'] for sample in test_dataset.samples]
        test_counter = Counter(test_labels)
        
        # 计算每个类别的数量
        train_class_counts = {cls: train_counter.get(cls, 0) for cls in range(config.num_classes)}
        test_class_counts = {cls: test_counter.get(cls, 0) for cls in range(config.num_classes)}
        
        # 计算每个类别的比例
        train_total = len(train_labels)
        train_class_ratios = {cls: count / train_total if train_total > 0 else 0.0 for cls, count in train_class_counts.items()}
        
        test_total = len(test_labels)
        test_class_ratios = {cls: count / test_total if test_total > 0 else 0.0 for cls, count in test_class_counts.items()}
        
        # 保存当前fold的分布
        fold_distributions.append({
            'fold': i + 1,
            'test_subject': test_subject,
            'train_size': train_total,
            'test_size': test_total,
            'train_distribution': train_class_counts,
            'test_distribution': test_class_counts,
            'train_ratios': train_class_ratios,
            'test_ratios': test_class_ratios
        })
        
        # 打印当前fold的分布
        log(f"  训练集类别分布:")
        for cls in range(config.num_classes):
            count = train_class_counts[cls]
            ratio = train_class_ratios[cls]
            log(f"    {class_names.get(cls, f'类别{cls}')}: {count} ({ratio:.2%})")
        
        log(f"  测试集类别分布:")
        for cls in range(config.num_classes):
            count = test_class_counts[cls]
            ratio = test_class_ratios[cls]
            log(f"    {class_names.get(cls, f'类别{cls}')}: {count} ({ratio:.2%})")
        
        # 检查是否有类别在测试集中为0
        zero_classes = [cls for cls in range(config.num_classes) if test_class_counts[cls] == 0]
        if zero_classes:
            log(f"  警告: 测试集中缺少以下类别: {[class_names.get(cls, f'类别{cls}') for cls in zero_classes]}")
        
        log("-" * 50)
    
    # 计算总体统计
    if fold_distributions:
        log("=" * 60)
        log("总体统计结果")
        log("=" * 60)
        
        # 计算每个类别的平均分布
        total_train_counts = {cls: 0 for cls in range(config.num_classes)}
        total_test_counts = {cls: 0 for cls in range(config.num_classes)}
        total_folds = len(fold_distributions)
        
        for fold in fold_distributions:
            for cls in range(config.num_classes):
                total_train_counts[cls] += fold['train_distribution'][cls]
                total_test_counts[cls] += fold['test_distribution'][cls]
        
        # 计算平均值
        avg_train_counts = {cls: count / total_folds for cls, count in total_train_counts.items()}
        avg_test_counts = {cls: count / total_folds for cls, count in total_test_counts.items()}
        
        log(f"平均每个fold的训练集大小: {np.mean([fold['train_size'] for fold in fold_distributions]):.1f}")
        log(f"平均每个fold的测试集大小: {np.mean([fold['test_size'] for fold in fold_distributions]):.1f}")
        
        log("\n平均训练集类别分布:")
        for cls in range(config.num_classes):
            count = avg_train_counts[cls]
            log(f"  {class_names.get(cls, f'类别{cls}')}: {count:.1f}")
        
        log("\n平均测试集类别分布:")
        for cls in range(config.num_classes):
            count = avg_test_counts[cls]
            log(f"  {class_names.get(cls, f'类别{cls}')}: {count:.1f}")
        
        # 统计有多少fold的测试集缺少类别
        folds_with_missing_classes = 0
        for fold in fold_distributions:
            zero_classes = [cls for cls in range(config.num_classes) if fold['test_distribution'][cls] == 0]
            if zero_classes:
                folds_with_missing_classes += 1
        
        log(f"\n有 {folds_with_missing_classes}/{total_folds} 个fold的测试集缺少至少一个类别")
        
        log("=" * 60)
    else:
        log("未分析到任何有效fold")

if __name__ == "__main__":
    analyze_fold_distribution()