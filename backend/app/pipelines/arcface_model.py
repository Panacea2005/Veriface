"""ArcFace ResNet-100 model architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        # Only create downsample when needed (channel mismatch or stride != 1)
        # This matches InsightFace iResNet implementation
        if in_channel == depth and stride == 1:
            # No downsample needed - use identity shortcut
            self.downsample = None
        else:
            # Need downsample: create Sequential with conv + bn
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        # Match checkpoint structure: bn1, conv1, bn2, prelu, conv2, bn3 (flat, not Sequential)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False)
        self.bn2 = nn.BatchNorm2d(depth)
        self.prelu = nn.PReLU(depth)
        self.conv2 = nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(depth)

    def forward(self, x):
        # Use downsample if exists, otherwise identity shortcut
        shortcut = self.downsample(x) if self.downsample is not None else x
        res = self.bn1(x)
        res = self.conv1(res)
        res = self.bn2(res)
        res = self.prelu(res)
        res = self.conv2(res)
        res = self.bn3(res)
        return res + shortcut

class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.downsample = nn.MaxPool2d(1, stride)
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        # Match checkpoint structure: flat layers
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False)
        self.bn2 = nn.BatchNorm2d(depth)
        self.prelu = nn.PReLU(depth)
        self.conv2 = nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(depth)
        self.se_module = SEModule(depth, 16)

    def forward(self, x):
        shortcut = self.downsample(x)
        res = self.bn1(x)
        res = self.conv1(res)
        res = self.bn2(res)
        res = self.prelu(res)
        res = self.conv2(res)
        res = self.bn3(res)
        res = self.se_module(res)
        return res + shortcut

def get_block(in_channel, depth, num_units, stride=2):
    return [bottleneck_IR(in_channel, depth, stride)] + [bottleneck_IR(depth, depth, 1) for i in range(num_units - 1)]

class Backbone(nn.Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        units = {
            50: [3, 4, 14, 3],
            100: [3, 13, 30, 3],
            152: [3, 8, 36, 3]
        }
        block = {
            'ir': bottleneck_IR,
            'ir_se': bottleneck_IR_SE
        }[mode]
        
        # Match checkpoint structure: conv1, bn1, prelu
        self.conv1 = nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        
        # Match checkpoint structure: layer1, layer2, layer3, layer4
        self.layer1 = nn.Sequential(*get_block(64, 64, units[num_layers][0], stride=1))
        self.layer2 = nn.Sequential(*get_block(64, 128, units[num_layers][1], stride=2))
        self.layer3 = nn.Sequential(*get_block(128, 256, units[num_layers][2], stride=2))
        self.layer4 = nn.Sequential(*get_block(256, 512, units[num_layers][3], stride=2))
        
        # Match checkpoint structure: fc (25088 -> 512) and features (BatchNorm1d)
        # 25088 = 512 * 7 * 7 (after avg pool and flatten)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.4)  # Match typical ArcFace dropout rate
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = nn.BatchNorm1d(512)

    def forward(self, x):
        # Match checkpoint forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.bn2(x)
        # AdaptiveAvgPool2d to ensure 7x7 output regardless of input size
        x = F.adaptive_avg_pool2d(x, (7, 7))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)  # Move dropout after flatten, before fc (common in ArcFace)
        x = self.fc(x)
        x = self.features(x)
        return x

def get_model(input_size=[112, 112], num_layers=100, mode='ir'):
    """Get ArcFace ResNet model."""
    model = Backbone(input_size, num_layers, mode)
    return model

