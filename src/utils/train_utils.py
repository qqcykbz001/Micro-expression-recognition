import torch
import torch.nn as nn
import numpy as np
import time
from torch.amp import autocast, GradScaler
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

class FocalLoss(nn.Module):
    """Focal Loss，兼容硬标签和Mixup软标签"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if isinstance(self.alpha, list):
            alpha = torch.tensor(self.alpha, device=inputs.device)
            if targets.dim() == 2:
                # Mixup软标签: [B, C]，按比例混合各类别的alpha
                alpha_weight = (targets * alpha).sum(dim=1)
            else:
                # 硬标签: [B]，直接索引
                alpha_weight = alpha[targets]
        else:
            alpha_weight = self.alpha

        focal_loss = alpha_weight * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train(model, train_loader, criterion, optimizer, device,
          accumulation_steps=1, use_amp=True, log_func=print,
          use_mixup=False, mixup_alpha=0.2, grad_clip_norm=1.0,
          num_classes=3):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = len(train_loader)

    if use_amp:
        scaler = GradScaler()

    with tqdm(total=batch_count, desc='Training', unit='batch') as pbar:
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            start_time = time.time()

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # --- Mixup ---
            if use_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                lam = max(lam, 1 - lam)  # 保证主要成分 > 0.5
                index = torch.randperm(inputs.size(0), device=device)

                # 图像混合
                inputs = lam * inputs + (1 - lam) * inputs[index]

                # 标签混合为软标签
                targets_onehot = torch.zeros(targets.size(0), num_classes, device=device)
                targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
                targets_b = targets_onehot[index]
                targets_mixed = lam * targets_onehot + (1 - lam) * targets_b

                loss_targets = targets_mixed
            else:
                loss_targets = targets

            if use_amp:
                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, loss_targets)
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    # 梯度裁剪
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, loss_targets)
                loss = loss / accumulation_steps
                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    # 梯度裁剪
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            batch_time = time.time() - start_time

            # 准确率用硬标签计算（Mixup时也用原始targets）
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            batch_accuracy = 100. * correct / total
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{batch_accuracy:.2f}%',
                'Time': f'{batch_time:.2f}s'
            })
            pbar.update(1)

    # 处理最后不完整的 accumulation step
    if (batch_idx + 1) % accumulation_steps != 0:
        if use_amp:
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()

    epoch_accuracy = 100. * correct / total
    epoch_loss = running_loss / batch_count
    return epoch_loss, epoch_accuracy


def test(model, test_loader, criterion, device, log_func=print):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_count = len(test_loader)

    all_targets = []
    all_predicted = []

    with torch.no_grad():
        with tqdm(total=batch_count, desc='Testing', unit='batch') as pbar:
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())

                batch_accuracy = 100. * correct / total
                avg_loss = test_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{batch_accuracy:.2f}%'
                })
                pbar.update(1)

    if total == 0:
        accuracy = 0.0
        avg_loss = 0.0
        uar = 0.0
        uf1 = 0.0
        log_func(f'  [Test Summary] Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, '
                 f'UAR: {uar:.2f}%, UF1: {uf1:.2f}% (Empty Test Set)')
    else:
        accuracy = 100. * correct / total
        avg_loss = test_loss / batch_count
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predicted, average='macro', zero_division=0)
        uar = recall * 100
        uf1 = f1 * 100
        log_func(f'  [Test Summary] Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, '
                 f'UAR: {uar:.2f}%, UF1: {uf1:.2f}%')

    return accuracy, uar, uf1, all_targets, all_predicted
