import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 训练可视化函数
def plot_training_metrics(train_losses, train_accuracies, test_accuracies, test_uar_scores, test_uf1s, learning_rates, fold, model_dir):
    """生成训练过程的可视化图表"""
    epochs = range(1, len(train_losses) + 1)
    
    # 设置绘图风格
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')
    
    # 创建一个包含2x2子图的图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Training Metrics - Fold {fold}', fontsize=20, fontweight='bold')
    
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
    
    # 调整子图间距
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图表
    os.makedirs(model_dir, exist_ok=True)
    plot_path = os.path.join(model_dir, f'training_metrics_fold{fold}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f'Training metrics plot saved to {plot_path}')

def plot_confusion_matrix(all_targets, all_predicted, classes, model_dir, title='Overall Confusion Matrix'):
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
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # 保存图表
    os.makedirs(model_dir, exist_ok=True)
    cm_path = os.path.join(model_dir, 'overall_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to {cm_path}')
