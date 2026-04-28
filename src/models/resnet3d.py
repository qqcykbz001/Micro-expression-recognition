import torch
import torch.nn as nn

class BasicBlock3D(nn.Module):
    """3D ResNet的基本残差块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_batch_norm=True):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()

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
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

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
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.BatchNorm3d(in_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        q = self.query(x).view(batch_size, -1, depth * height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, depth * height * width)
        v = self.value(x).view(batch_size, -1, depth * height * width)
        attention = self.softmax(torch.bmm(q, k) / (self.in_channels ** 0.5))
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, depth, height, width)
        out = self.norm(out + x)
        out = self.relu(out)
        return out

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM) — 3D extension"""
    def __init__(self, in_channels, reduction=8):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_out = self.channel_attention(x)
        x = x * channel_out
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.spatial_attention(spatial_in)
        x = x * spatial_out
        return x

class ResNet3D(nn.Module):
    """3D ResNet模型"""
    def __init__(self, block, layers, num_classes=10, use_attention=True, attention_type='cbam',
                 use_dropout=True, dropout_rate=0.5, input_channels=3, use_batch_norm=True, config=None):
        super(ResNet3D, self).__init__()
        self.use_attention = use_attention
        self.attention_type = attention_type
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.input_channels = input_channels
        self.config = config

        self.use_two_stream = input_channels == 4

        if self.use_two_stream:
            # 灰度流分支
            self.gray_conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                                        padding=(1, 3, 3), bias=False)
            self.gray_bn1 = nn.BatchNorm3d(64) if use_batch_norm else nn.Identity()
            self.gray_relu = nn.ReLU(inplace=True)
            self.gray_layer1 = self._make_layer(block, 64, layers[0], stride=1, in_channels=64)
            self.gray_layer2 = self._make_layer(block, 128, layers[1], stride=2, in_channels=64 * block.expansion)
            self.gray_layer3 = self._make_layer(block, 256, layers[2], stride=2, in_channels=128 * block.expansion)
            self.gray_layer4 = self._make_layer(block, 512, layers[3], stride=2, in_channels=256 * block.expansion)
            self.gray_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

            # 光流分支
            self.flow_conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                                        padding=(1, 3, 3), bias=False)
            self.flow_bn1 = nn.BatchNorm3d(64) if use_batch_norm else nn.Identity()
            self.flow_relu = nn.ReLU(inplace=True)
            self.flow_layer1 = self._make_layer(block, 64, layers[0], stride=1, in_channels=64)
            self.flow_layer2 = self._make_layer(block, 128, layers[1], stride=2, in_channels=64 * block.expansion)
            self.flow_layer3 = self._make_layer(block, 256, layers[2], stride=2, in_channels=128 * block.expansion)
            self.flow_layer4 = self._make_layer(block, 512, layers[3], stride=2, in_channels=256 * block.expansion)
            self.flow_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

            if self.use_attention:
                if attention_type == 'cbam':
                    self.gray_attention = CBAM(512 * block.expansion)
                    self.flow_attention = CBAM(512 * block.expansion)
                else:
                    self.gray_attention = AttentionBlock(512 * block.expansion, use_batch_norm=use_batch_norm)
                    self.flow_attention = AttentionBlock(512 * block.expansion, use_batch_norm=use_batch_norm)

            # Gate 融合
            gate_in = 512 * block.expansion * 2
            gate_bottleneck = gate_in // 8
            self.gate_fc = nn.Sequential(
                nn.Linear(gate_in, gate_bottleneck),
                nn.ReLU(),
                nn.Linear(gate_bottleneck, gate_in),
                nn.Sigmoid()
            )

            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            # 单流法
            self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                                   padding=(1, 3, 3), bias=False)
            self.bn1 = nn.BatchNorm3d(64) if use_batch_norm else nn.Identity()
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(block, 64, layers[0], stride=1, in_channels=64)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, in_channels=64 * block.expansion)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, in_channels=128 * block.expansion)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, in_channels=256 * block.expansion)

            if self.use_attention:
                if attention_type == 'cbam':
                    self.attention = CBAM(512 * block.expansion)
                else:
                    self.attention = AttentionBlock(512 * block.expansion, use_batch_norm=use_batch_norm)

            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.use_dropout:
            self.dropout = nn.Dropout(p=self.dropout_rate)

    def _make_layer(self, block, out_channels, blocks, stride, in_channels=None):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        current_in_channels = in_channels
        for s in strides:
            layers.append(block(current_in_channels, out_channels, s, use_batch_norm=self.use_batch_norm))
            current_in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_two_stream:
            gray_stream = x[:, :1].repeat(1, 3, 1, 1, 1)
            flow_stream = x[:, 1:]

            gray_feat = self.gray_relu(self.gray_bn1(self.gray_conv1(gray_stream)))
            gray_feat = self.gray_layer1(gray_feat)
            gray_feat = self.gray_layer2(gray_feat)
            gray_feat = self.gray_layer3(gray_feat)
            gray_feat = self.gray_layer4(gray_feat)
            if self.use_attention:
                gray_feat = self.gray_attention(gray_feat)
            gray = self.gray_avgpool(gray_feat).flatten(1)

            flow_feat = self.flow_relu(self.flow_bn1(self.flow_conv1(flow_stream)))
            flow_feat = self.flow_layer1(flow_feat)
            flow_feat = self.flow_layer2(flow_feat)
            flow_feat = self.flow_layer3(flow_feat)
            flow_feat = self.flow_layer4(flow_feat)
            if self.use_attention:
                flow_feat = self.flow_attention(flow_feat)
            flow = self.flow_avgpool(flow_feat).flatten(1)

            gate_input = torch.cat([gray, flow], dim=1)
            gate_weights = self.gate_fc(gate_input)
            gray_w, flow_w = gate_weights[:, :gray.size(1)], gate_weights[:, gray.size(1):]
            x = gray * gray_w + flow * flow_w
        else:
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1, 1)

            feat = self.relu(self.bn1(self.conv1(x)))
            feat = self.layer1(feat)
            feat = self.layer2(feat)
            feat = self.layer3(feat)
            feat = self.layer4(feat)

            if self.use_attention:
                feat = self.attention(feat)

            x = self.avgpool(feat).flatten(1)

        if self.use_dropout:
            x = self.dropout(x)

        x = self.fc(x)
        return x


def resnet3d18(num_classes=10, use_attention=True, attention_type='cbam',
               use_dropout=True, dropout_rate=0.5, input_channels=3,
               use_batch_norm=True, config=None):
    """3D ResNet-18模型"""
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes, use_attention,
                    attention_type, use_dropout, dropout_rate, input_channels,
                    use_batch_norm, config)


def resnet3d34(num_classes=10, use_attention=True, attention_type='cbam',
               use_dropout=True, dropout_rate=0.5, input_channels=3,
               use_batch_norm=True, config=None):
    """3D ResNet-34模型"""
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], num_classes, use_attention,
                    attention_type, use_dropout, dropout_rate, input_channels,
                    use_batch_norm, config)
