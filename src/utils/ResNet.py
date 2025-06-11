import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = nn.Conv2d(self.in_planes, self.planes, kernel_size=3, stride=self.stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.planes)
        self.conv2 = nn.Conv2d(self.planes, self.planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, self.expansion * self.planes, kernel_size=1,
                          stride=self.stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.planes)
            )

    def forward(self, x):
        x_out = F.relu(self.bn1(self.conv1(x)))
        x_out = self.bn2(self.conv2(x_out))
        x_out += self.shortcut(x)
        x_out = F.relu(x_out)

        return x_out
    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet, self).__init__()
        self.block = block
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers of the network using _make_layers
        self.layer1 = self._make_layers(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layers(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layers(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layers(self.block, 512, self.num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * self.block.expansion, self.num_classes)
        

    def _make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

