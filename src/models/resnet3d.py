import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# 日志文件路径
LOG_FILE = "training.log"

# 日志写入函数
def log(message):
    """将日志写入文件"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message)

class BasicBlock3D(nn.Module):
    """3D ResNet的基本残差块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_batch_norm=True):
        super(BasicBlock3D, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()
        
        #  shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()
            )
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class Bottleneck3D(nn.Module):
    """3D ResNet的瓶颈残差块"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, use_batch_norm=True):
        super(Bottleneck3D, self).__init__()
        # 1x1x1卷积降维
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()
        # 3x3x3卷积处理
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()
        # 1x1x1卷积升维
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
        # shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * self.expansion) if use_batch_norm else nn.Identity()
            )
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    """自注意力模块"""
    def __init__(self, in_channels, use_batch_norm=True):
        super(AttentionBlock, self).__init__()
        self.in_channels = in_channels
        
        # 自注意力机制
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
        # 残差连接
        self.norm = nn.BatchNorm3d(in_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        
        # 计算query, key, value
        q = self.query(x).view(batch_size, -1, depth * height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, depth * height * width)
        v = self.value(x).view(batch_size, -1, depth * height * width)
        
        # 计算注意力权重
        attention = self.softmax(torch.bmm(q, k) / (self.in_channels ** 0.5))
        
        # 计算注意力加权的特征
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, depth, height, width)
        
        # 残差连接
        out = self.norm(out + x)
        out = self.relu(out)
        
        return out

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # 全局平均池化
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False),  # 7x7卷积
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        channel_out = self.channel_attention(x)
        x = x * channel_out
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.spatial_attention(spatial_in)
        x = x * spatial_out
        
        return x

class ResNet3D(nn.Module):
    """3D ResNet模型"""
    def __init__(self, block, layers, num_classes=10, use_attention=True, attention_type='cbam', use_dropout=True, dropout_rate=0.5, input_channels=3, use_batch_norm=True):
        super(ResNet3D, self).__init__()
        self.in_channels = 64
        self.use_attention = use_attention
        self.attention_type = attention_type
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # 初始卷积层
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 添加注意力模块
        if self.use_attention:
            if attention_type == 'cbam':
                self.attention = CBAM(512 * block.expansion)
            else:
                self.attention = AttentionBlock(512 * block.expansion, use_batch_norm=use_batch_norm)
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 添加dropout层
        if self.use_dropout:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, use_batch_norm=self.use_batch_norm))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # 使用dropout层
        if self.use_dropout:
            x = self.dropout(x)
        
        x = self.fc(x)
        return x

# 预定义不同深度的3D ResNet模型
def resnet3d18(num_classes=10, use_attention=True, attention_type='cbam', pretrained=False, use_dropout=True, dropout_rate=0.5, input_channels=3, use_batch_norm=True):
    """3D ResNet-18模型"""
    model = ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes, use_attention, attention_type, use_dropout, dropout_rate, input_channels, use_batch_norm)
    if pretrained:
        try:
            # 尝试使用PyTorch Video库加载预训练权重
            from pytorchvideo.models import create_resnet
            log("从PyTorch Video加载预训练权重...")
            # 创建预训练模型
            pretrained_model = create_resnet(
                input_channel=3,
                model_depth=18,
                model_num_class=400,  # Kinetics-400
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )
            # 复制权重到我们的模型
            pretrained_dict = pretrained_model.state_dict()
            model_dict = model.state_dict()
            # 过滤掉不匹配的层
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 更新模型权重
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            log("预训练权重从PyTorch Video加载成功")
        except ImportError:
            log("PyTorch Video未安装。使用随机初始化。")
        except Exception as e:
            log(f"加载预训练权重时出错: {e}")
            log("使用随机初始化。")
    return model

def resnet3d34(num_classes=10, use_attention=True, attention_type='cbam', pretrained=False, use_dropout=True, dropout_rate=0.5, input_channels=3, use_batch_norm=True):
    """3D ResNet-34模型"""
    model = ResNet3D(BasicBlock3D, [3, 4, 6, 3], num_classes, use_attention, attention_type, use_dropout, dropout_rate, input_channels, use_batch_norm)
    if pretrained:
        try:
            # 尝试使用PyTorch Video库加载预训练权重
            from pytorchvideo.models import create_resnet
            log("从PyTorch Video加载预训练权重...")
            # 创建预训练模型
            pretrained_model = create_resnet(
                input_channel=3,
                model_depth=34,
                model_num_class=400,  # Kinetics-400
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )
            # 复制权重到我们的模型
            pretrained_dict = pretrained_model.state_dict()
            model_dict = model.state_dict()
            # 过滤掉不匹配的层
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 更新模型权重
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            log("预训练权重从PyTorch Video加载成功")
        except ImportError:
            log("PyTorch Video未安装。使用随机初始化。")
        except Exception as e:
            log(f"加载预训练权重时出错: {e}")
            log("使用随机初始化。")
    return model

def resnet3d50(num_classes=10, use_attention=True, attention_type='cbam', pretrained=False, use_dropout=True, dropout_rate=0.5, input_channels=3, use_batch_norm=True):
    """3D ResNet-50模型"""
    model = ResNet3D(Bottleneck3D, [3, 4, 6, 3], num_classes, use_attention, attention_type, use_dropout, dropout_rate, input_channels, use_batch_norm)
    if pretrained:
        try:
            # 尝试使用PyTorch Video库加载预训练权重
            from pytorchvideo.models import create_resnet
            log("从PyTorch Video加载预训练权重...")
            # 创建预训练模型
            pretrained_model = create_resnet(
                input_channel=3,
                model_depth=50,
                model_num_class=400,  # Kinetics-400
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )
            # 复制权重到我们的模型
            pretrained_dict = pretrained_model.state_dict()
            model_dict = model.state_dict()
            # 过滤掉不匹配的层
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 更新模型权重
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            log("预训练权重从PyTorch Video加载成功")
        except ImportError:
            log("PyTorch Video未安装。使用随机初始化。")
        except Exception as e:
            log(f"加载预训练权重时出错: {e}")
            log("使用随机初始化。")
    return model

def resnet3d101(num_classes=10, use_attention=True, attention_type='cbam', pretrained=False, use_dropout=True, dropout_rate=0.5, input_channels=3, use_batch_norm=True):
    """3D ResNet-101模型"""
    model = ResNet3D(Bottleneck3D, [3, 4, 23, 3], num_classes, use_attention, attention_type, use_dropout, dropout_rate, input_channels, use_batch_norm)
    if pretrained:
        try:
            # 尝试使用PyTorch Video库加载预训练权重
            from pytorchvideo.models import create_resnet
            log("从PyTorch Video加载预训练权重...")
            # 创建预训练模型
            pretrained_model = create_resnet(
                input_channel=3,
                model_depth=101,
                model_num_class=400,  # Kinetics-400
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )
            # 复制权重到我们的模型
            pretrained_dict = pretrained_model.state_dict()
            model_dict = model.state_dict()
            # 过滤掉不匹配的层
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 更新模型权重
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            log("预训练权重从PyTorch Video加载成功")
        except ImportError:
            log("PyTorch Video未安装。使用随机初始化。")
        except Exception as e:
            log(f"加载预训练权重时出错: {e}")
            log("使用随机初始化。")
    return model

def resnet3d152(num_classes=10, use_attention=True, attention_type='cbam', pretrained=False, use_dropout=True, dropout_rate=0.5, input_channels=3, use_batch_norm=True):
    """3D ResNet-152模型"""
    model = ResNet3D(Bottleneck3D, [3, 8, 36, 3], num_classes, use_attention, attention_type, use_dropout, dropout_rate, input_channels, use_batch_norm)
    if pretrained:
        try:
            # 尝试使用PyTorch Video库加载预训练权重
            from pytorchvideo.models import create_resnet
            log("从PyTorch Video加载预训练权重...")
            # 创建预训练模型
            pretrained_model = create_resnet(
                input_channel=3,
                model_depth=152,
                model_num_class=400,  # Kinetics-400
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )
            # 复制权重到我们的模型
            pretrained_dict = pretrained_model.state_dict()
            model_dict = model.state_dict()
            # 过滤掉不匹配的层
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 更新模型权重
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            log("预训练权重从PyTorch Video加载成功")
        except ImportError:
            log("PyTorch Video未安装。使用随机初始化。")
        except Exception as e:
            log(f"加载预训练权重时出错: {e}")
            log("使用随机初始化。")
    return model
