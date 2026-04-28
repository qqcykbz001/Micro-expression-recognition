import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 训练可视化函数
def plot_training_metrics(train_losses, train_accuracies, test_accuracies, test_uar_scores, test_uf1s, learning_rates, fold, model_dir, dataset_name='casme2', late_select_start=None):
    """生成训练过程的可视化图表"""
    epochs = range(1, len(train_losses) + 1)
    
    # 设置绘图风格
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')
    
    # 创建一个包含2x2子图的图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Training Metrics - {dataset_name} - Fold {fold}', fontsize=20, fontweight='bold')
    
    # 1. 损失曲线 (训练损失)
    axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[0, 0].set_title('Loss Curve', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epochs', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 2. 准确率曲线 (训练 vs 测试)
    axes[0, 1].plot(epochs, train_accuracies, 'g-', linewidth=2, label='Train Acc')
    axes[0, 1].plot(epochs, test_accuracies, 'r--', linewidth=2, label='Test Acc')
    axes[0, 1].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epochs', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # 3. 核心指标 (UAR vs UF1)
    axes[1, 0].plot(epochs, test_uar_scores, 'orange', linewidth=2, label='Test UAR')
    axes[1, 0].plot(epochs, test_uf1s, 'purple', linewidth=2, label='Test UF1')
    axes[1, 0].set_title('UAR & UF1 Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epochs', fontsize=12)
    axes[1, 0].set_ylabel('Score (%)', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 4. 学习率变化
    axes[1, 1].plot(epochs, learning_rates, 'y-', linewidth=2, label='Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epochs', fontsize=12)
    axes[1, 1].set_ylabel('LR', fontsize=12)
    axes[1, 1].set_yscale('log') # 使用对数刻度更直观
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    # 标记 late-phase 起始线
    if late_select_start is not None and late_select_start <= len(epochs):
        for ax in axes.flatten():
            ax.axvline(late_select_start, color='gray', ls=':', lw=1.2, alpha=0.7)

    # 调整子图间距
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图表
    os.makedirs(model_dir, exist_ok=True)
    plot_path = os.path.join(model_dir, f'training_metrics_{dataset_name}_fold{fold}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f'Training metrics plot saved to {plot_path}')

def plot_confusion_matrix(all_targets, all_predicted, classes, model_dir, title='Overall Confusion Matrix', dataset_name='casme2'):
    """生成并保存混淆矩阵"""
    cm = confusion_matrix(all_targets, all_predicted)
    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(font_scale=1.2)
    
    # 使用热力图绘制，同时显示原始数值和百分比
    labels = (np.array(["{0:d}\n({1:.1%})".format(count, pct) 
              for count, pct in zip(cm.flatten(), cm_normalized.flatten())])).reshape(cm.shape)
    
    sns.heatmap(cm, annot=labels, fmt="", cmap='Blues', xticklabels=classes, yticklabels=classes)
    
    plt.title(f'{title} - {dataset_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # 保存图表
    os.makedirs(model_dir, exist_ok=True)
    cm_path = os.path.join(model_dir, f'overall_confusion_matrix_{dataset_name}.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to {cm_path}')


def plot_loso_summary(accuracies, uar_scores, uf1_scores, valid_subjects,
                      model_dir, dataset_name='casme2'):
    """LOSO 汇总柱状图：每 fold 的 Acc/UAR/UF1 + 均值线 + 统计文本"""
    n = len(accuracies)
    if n == 0:
        return
    x = np.arange(n)
    w = 0.25

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(max(12, n * 0.45), 6))
    ax.bar(x - w, accuracies, w, color='steelblue', edgecolor='white', label='Acc')
    ax.bar(x, uar_scores, w, color='darkorange', edgecolor='white', label='UAR')
    ax.bar(x + w, uf1_scores, w, color='mediumorchid', edgecolor='white', label='UF1')

    for vals, color in [(accuracies, 'steelblue'), (uar_scores, 'darkorange'), (uf1_scores, 'mediumorchid')]:
        ax.axhline(np.mean(vals), color=color, ls='--', lw=1.5, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(valid_subjects, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('%')
    ax.set_title(f'LOSO Cross-Validation — {dataset_name.upper()}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', ls='--', alpha=0.5)

    mean_acc, std_acc = np.mean(accuracies), np.std(accuracies)
    mean_uar, std_uar = np.mean(uar_scores), np.std(uar_scores)
    mean_uf1, std_uf1 = np.mean(uf1_scores), np.std(uf1_scores)
    summary = (
        f"Mean Acc: {mean_acc:.2f}% ± {std_acc:.2f}%\n"
        f"Mean UAR: {mean_uar:.2f}% ± {std_uar:.2f}%\n"
        f"Mean UF1: {mean_uf1:.2f}% ± {std_uf1:.2f}%\n"
        f"Folds: {n}"
    )
    ax.text(0.98, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f'loso_summary_{dataset_name}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'LOSO summary saved to {path}')