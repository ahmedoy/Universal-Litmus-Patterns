import torch
import torch.nn as nn
import torch.nn.functional as F

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        return torch.cat([x, out], 1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            layers.append(_DenseLayer(channels, growth_rate, bn_size))
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels
        
    def forward(self, x):
        return self.block(x)


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)
        
    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


class DenseNetCIFAR(nn.Module):
    architecture_name = "densenet"
    def __init__(self, growth_rate=12, block_layers=(6, 6, 6), compression=0.5, num_classes=10):
        super().__init__()
        # Initial convolution
        num_init_features = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=3, padding=1, bias=False)
        
        # Dense blocks + transitions
        channels = num_init_features
        self.blocks = nn.ModuleList()
        for i, num_layers in enumerate(block_layers):
            # Dense block
            block = _DenseBlock(num_layers, channels, growth_rate)
            self.blocks.append(block)
            channels = block.out_channels
            
            # Add transition layer except after last block
            if i != len(block_layers) - 1:
                out_channels = int(channels * compression)
                trans = _Transition(channels, out_channels)
                self.blocks.append(trans)
                channels = out_channels
        
        # Final batch norm
        self.norm_final = nn.BatchNorm2d(channels)
        
        # Classification layer
        self.classifier = nn.Linear(channels, num_classes)
        
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        for layer in self.blocks:
            x = layer(x)
        x = F.relu(self.norm_final(x))
        # global average pooling
        x = F.adaptive_avg_pool2d(x, (1,1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x