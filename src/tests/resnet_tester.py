import torch
from utils.ResNet import ResNet, BasicBlock

def ResNet18(num_classes=1000):
    resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return resnet

if __name__ == "__main__":
    resnet = ResNet18()
    print(resnet)

    # dummy tensor
    x = torch.randn(1, 3 ,64, 64)
    output = resnet(x)
    print(f"out shape: {output.shape}")