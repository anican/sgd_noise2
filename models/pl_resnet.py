import argparse
from models.pl_base import LightningBase
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # TODO: Remove self.shortcut if you can't find a use for it
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # TODO: nn.Sequential format instead!
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # TODO: Remove self.shortcut if you can't find a use for it
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # TODO: nn.Sequential format instead!
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(LightningBase):
    def __init__(self, hparams, paths, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__(hparams, paths)
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, get_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        if get_features:
            out_flat = out.view(out.size(0), -1)
            out = self.linear(out_flat)
            return out, out_flat
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(hparams, paths):
    num_blocks = [2, 2, 2, 2]
    return ResNet(hparams=hparams, paths=paths, block=BasicBlock, num_blocks=num_blocks, num_classes=10)


def ResNet34(hparams, paths):
    num_blocks = [3, 4, 6, 3]
    return ResNet(hparams=hparams, paths=paths, block=BasicBlock, num_blocks=num_blocks, num_classes=10)


def ResNet50(hparams, paths):
    num_blocks = [3, 4, 6, 3]
    return ResNet(hparams=hparams, paths=paths, block=Bottleneck, num_blocks=num_blocks, num_classes=10)


def ResNet101(hparams, paths):
    num_blocks = [3, 4, 23, 3]
    return ResNet(hparams=hparams, paths=paths, block=Bottleneck, num_blocks=num_blocks, num_classes=10)


def ResNet152(hparams, paths):
    num_blocks = [3, 8, 36, 3]
    return ResNet(hparams=hparams, paths=paths, block=Bottleneck, num_blocks=num_blocks, num_classes=10)


def _test_basic(args):
    # net = ResNet(hparams=args, paths="hi", num_blocks=[2,2,2,2], block=BasicBlock)
    model18 = ResNet18(args, paths="hi")
    y = model18(torch.randn(50, 3, 32, 32))
    print("resnet18:", "BasicBlock", tuple(y.size()))

    model34 = ResNet34(args, paths="hi")
    y = model34(torch.randn(50, 3, 32, 32))
    print("resnet34:", "BasicBlock", tuple(y.size()))


def _test_bottleneck(args):
    model50 = ResNet50(args, paths='')
    y = model50(torch.randn(50, 3, 32, 32))
    print("resnet50:", "Bottleneck", tuple(y.size()))


    model101 = ResNet101(args, paths='')
    y = model101(torch.randn(50, 3, 32, 32))
    print("resnet101:", "Bottleneck", tuple(y.size()))

    model152 = ResNet152(args, paths='')
    y = model152(torch.randn(50, 3, 32, 32))
    print("resnet152:", "Bottleneck", tuple(y.size()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='lightning_tuna')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    test_args = parser.parse_args()
    _test_basic(test_args)
    _test_bottleneck(test_args)

