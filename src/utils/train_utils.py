import torch
import torch.nn as nn
import time
from torch.amp import autocast, GradScaler
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

class FocalLoss(nn.Module):
    """加权 Focal Loss，支持标签平滑"""
    def __init__(self, alpha=1, gamma=2, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        if isinstance(self.alpha, list):
            alpha = torch.tensor(self.alpha, device=inputs.device)
            alpha = alpha[targets]
        else:
            alpha = self.alpha
        
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 训练函数
def train(model, train_loader, criterion, optimizer, device, accumulation_steps=2, use_amp=True, log_func=print):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = len(train_loader)
    
    # 混合精度训练
    if use_amp:
        scaler = GradScaler()
    
    # 使用tqdm创建进度条
    with tqdm(total=batch_count, desc='Training', unit='batch') as pbar:
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            start_time = time.time()
            
            # 启用 non_blocking=True 配合 pin_memory 提升性能
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            if use_amp:
                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            batch_time = time.time() - start_time
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            batch_accuracy = 100.*correct/total
            avg_loss = running_loss/(batch_idx+1)
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{batch_accuracy:.2f}%',
                'Time': f'{batch_time:.2f}s'
            })
            pbar.update(1)
    
    if (batch_idx + 1) % accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()
    
    epoch_accuracy = 100.*correct/total
    epoch_loss = running_loss/batch_count
    return epoch_loss, epoch_accuracy

# 测试函数
def test(model, test_loader, criterion, device, log_func=print):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_count = len(test_loader)
    
    all_targets = []
    all_predicted = []
    
    # 使用tqdm创建进度条
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
                
                # 更新进度条
                batch_accuracy = 100.*correct/total
                avg_loss = test_loss/(batch_idx+1)
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
        log_func(f'  [Test Summary] Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, UAR: {uar:.2f}%, UF1: {uf1:.2f}% (Empty Test Set)')
    else:
        accuracy = 100.*correct/total
        avg_loss = test_loss/batch_count
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predicted, average='macro', zero_division=0)
        uar = recall * 100
        uf1 = f1 * 100
        log_func(f'  [Test Summary] Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, UAR: {uar:.2f}%, UF1: {uf1:.2f}%')
    
    return accuracy, uar, uf1, all_targets, all_predicted