#!/usr/bin/env python3
"""分析每个数据集的全局类别分布，输出推荐 focal_alpha"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from collections import Counter
from src.configs.config import Config
from src.datasets import get_dataset


def analyze_dataset(ds_name):
    cfg = Config()
    cfg.dataset_name = ds_name
    cfg.root_dir = cfg.dataset_roots[ds_name]

    print(f"\n{'='*60}")
    print(f"  {ds_name.upper()}")
    print(f"{'='*60}")

    dataset = get_dataset(cfg, log_func=print)
    labels = [s['label'] for s in dataset.samples]
    counts = Counter(labels)
    total = len(labels)

    print(f"总样本: {total}")
    for cls in sorted(counts):
        print(f"  类别 {cls} ({dataset.class_names[cls]}): {counts[cls]}  ({counts[cls]/total:.1%})")

    # focal_alpha
    weights = [total / counts[i] if i in counts else 1.0 for i in range(cfg.num_classes)]
    mean_w = sum(weights) / len(weights)
    alpha = [w / mean_w for w in weights]

    print(f"\n推荐 focal_alpha: {[round(a, 4) for a in alpha]}")
    print(f"  (类别权重: {[round(w, 2) for w in weights]})")
    return alpha


if __name__ == "__main__":
    for ds in ["casme2", "samm"]:
        analyze_dataset(ds)
