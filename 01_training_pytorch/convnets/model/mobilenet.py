"""
    MobileNet-V2 from the paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    by Mark Sandler et al, Google, CVPR 2018
"""

# imports
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# mobilenet variants
__all__ = [
    'MobileNet', 'mobilenet20_1p0_t3', 'mobilenet20_1p0_t4'
]

# pretrained models
model_urls = {
    'mobilenet20': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


# 3x3 conv filter preserving fmap dimensions
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    3x3 conv filter
    - preserves fmap dimensions if stride=1
    - exactly halves fmap dimensions if stride=2
    - becomes depthwise convolution if in_planes=out_planes=groups

    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups,
                     dilation=dilation, padding=dilation, bias=False)

# 1x1 conv filter preserving fmap dimensions
def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    1x1 conv filter
    - preserves fmap dimensions if stride=1
    - exactly halves fmap dimensions if stride=2

    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups,
                     dilation=dilation, padding=0, bias=False)

class MBConv(nn.Module):
    """
    mobile-inverted-bottleneck block

    structure:
    - 1x1-conv > bn > relu (input = h x w x k; output = h x w x t*k)
    - 3x3-conv-dw, stride=1/2 > bn > relu (output = h/s x w/s x t*k)
    - 1x1-conv > bn (output = h/s x w/s x k')
    - skip connection
    parameters:
    - t: expansion factor
    - k: input channels
    - k': output channels
    - s: stride
    notes:
    - in the original paper used ReLU6 activation
    - last 1x1-conv layer is linear, no activation function
    """

    def __init__(self, inplanes, outplanes, expansion=1, stride=1, dropout=0, downsample=None,
                 norm_layer=None, activation=None):
        """
        Constructor

        Args:
            inplanes: (int) number of input channels to block
            outplanes: (int) number of output channels from block
            expansion: (float) expansion factor t
            stride: (int) strides applied to intermediate 3x3-conv-dw layer
            dropout: (float) p = dropout for nn.dropout; default=0 returns identity (no effect on the network)
            downsample: (nn.Module) downsamples input tensor for skip connection; must set if stride > 1
            norm_layer: (nn.Module) normalization layer; default = nn.BatchNorm2d
            actiavtion: (nn.Module) activation layer; default = nn.ReLU6
        """
        super(MBConv, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if activation is None:
            activation = nn.ReLU6

        self._bottleneck_planes = int(inplanes * expansion)

        self.conv1 = conv1x1(inplanes, self._bottleneck_planes, stride=1)
        self.bn1 = norm_layer(self._bottleneck_planes)
        self.relu = activation(inplace=True)
        self.convdw2 = conv3x3(self._bottleneck_planes, self._bottleneck_planes, stride=stride,
                               groups=self._bottleneck_planes)
        self.bn2 = norm_layer(self._bottleneck_planes)
        self.conv3 = conv1x1(self._bottleneck_planes, outplanes, stride=1)
        self.bn3 = norm_layer(outplanes)

        self.downsample = downsample
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """ forward method """

        identity = x
        # downsamples if stride > 1
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.convdw2(out)))
        out = self.dropout(out)
        # last conv layer is linear
        out = self.bn3(self.conv3(out))
        out = self.dropout(out)

        # skip connection
        out += identity
        return out


class MobileNet(nn.Module):
    """
    MobileNet-V2

    Structure:

        ImageNet
        - 3x3-conv s=2
        - MBConv x1 s=1 t=1
        - MBConv x2 s=2 t=6
        - MBConv x3 s=2 t=6
        - MBConv x4 s=2 t=6
        - MBConv x3 s=1 t=6
        - MBConv x3 s=2 t=6
        - MBConv x1 s=1 t=6
        - 1x1-conv s=1 > bn > relu > avg pool > 1x1-conv

        Cifar-10/100
        - 3x3-conv s=1
        - MBConv x1 s=1 t=1
        - MBConv x2 s=2 t=6
        - MBConv x3 s=2 t=6
        - MBConv x4 s=1 t=6
        - MBConv x3 s=1 t=6
        - MBConv x3 s=2 t=6
        - MBConv x1 s=1 t=6
        - 1x1-conv s=1 > bn > relu > avg pool > 1x1-conv

    notes:
    - follows ResNet design principle for the progression of block output channels?
        - if fmap is downsampled by two within the block, then expand final block output by two
        - if no downsample, always maintain constant block output
        - all blocks in the same stack always share the equal output channels
    - only the first block in a stack could use stride=2; other blocks use stride=1

    """

    def __init__(self, block, layers, expansion=1, width_mult=1, num_classes=10,
                 dropout=0, norm_layer=None, activation=None):
        """
        Constructor

        Args:
            block: (nn.Module) building block for MobileNets; default is MBConv6
            layers: (list of integers) a list of several integers, each representing the number of blocks per stack
            expansion: (float) expansion factor for intermediate channels; default=1
            width_mult: (float) width multiplier; layer channels = original channels * width_mult
            num_classes: (int) number of classes
            dropout: (float) p = dropout; default = 0 (no effect)
            norm_layer: (nn.Module) normalization layer; default = nn.BatchNorm2d
            activation: (nn.Module) activation layer; default = nn.ReLU6
        """

        super(MobileNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer

        if activation is None:
            activation = nn.ReLU6
            self._activation = activation

        self.outplanes = int(16 * width_mult)

        self.conv1 = conv3x3(3, self.outplanes, stride=1)
        self.bn1 = norm_layer(self.outplanes)

        self.stack1 = self._make_stack(block, layers[0], self.outplanes, 16, expansion=expansion,
                                       stride=1, dropout=dropout)
        self.stack2 = self._make_stack(block, layers[1], 16, 32, expansion=expansion,
                                       stride=2, dropout=dropout)
        self.stack3 = self._make_stack(block, layers[2], 32, 64, expansion=expansion,
                                       stride=2, dropout=dropout)
        self.stack4 = self._make_stack(block, layers[3], 64, 96, expansion=expansion,
                                       stride=1, dropout=dropout)
        self.stack5 = self._make_stack(block, layers[4], 96, 128, expansion=expansion,
                                       stride=1, dropout=dropout)
        self.stack6 = self._make_stack(block, layers[5], 128, 256, expansion=expansion,
                                       stride=2, dropout=dropout)
        self.stack7 = self._make_stack(block, layers[6], 256, 384, expansion=expansion,
                                       stride=1, dropout=dropout)
        self.conv8 = conv1x1(384, 512, stride=1)
        self.bn8 = norm_layer(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv9 = conv1x1(512, num_classes, stride=1)

        self.relu = activation(inplace=True)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stack(self, block, num_layers, inplanes, outplanes, expansion=1, stride=1, dropout=0):
        """
        build a stack of blocks

        Args:
            block: (nn.Module) building block; default = MBConv6
            num_layers: (int) number of blocks/layers in the stack
            inplanes: (int) number of input channels to the first block of the stack
            outplanes: (int) number of output channels of the block; all blocks in the stack share equal outplanes value
            expansion: (float) expansion factor
            stride: (int) stride=1/2 for first block; stride=1 for other blocks
            dropout: (float) p = dropout; default=0 no dropout effect
        """

        norm_layer = self._norm_layer
        activation = self._activation
        downsample = None

        # if stride != 1
        # or if block input != block output (only possible for first block in the stack)
        # apply downsample on identity shortcut by 1x1-conv w.t. outplanes = block output channels, stride = block stride
        if stride != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                conv1x1(inplanes, outplanes, stride=stride),
                norm_layer(outplanes)
            )

        layers = []

        # first block in the stack; could use stride=2 plus downsample
        layers.append(block(inplanes, outplanes, expansion=expansion, stride=stride, dropout=dropout,
                            downsample=downsample, norm_layer=norm_layer, activation=activation))

        # other layers of block
        # fix stride=1, no downsample, inplanes = outplanes
        for _ in range(1, num_layers):
            layers.append(block(outplanes, outplanes, expansion=expansion, stride=1, dropout=dropout,
                                norm_layer=norm_layer, activation=activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        """ forward method """

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)
        x = self.stack5(x)
        x = self.stack6(x)
        x = self.stack7(x)
        x = self.relu(self.bn8(self.conv8(x)))

        x = self.avgpool(x)
        x = self.conv9(x)

        x = torch.flatten(x, 1)

        return x


def _mobilenet(arch, block, layers, width_mult=1, expansion=1, dropout=0, pretrained=False, progress=False, **kwargs):
    """
    Abstract generator function to build mobilenet variants

    Args:
        arch: (string) architecture of pretrained model
        block: (nn.Module) name of building block; default is MBConv6
        width_mult: (float) width multiplier
        expansion: (float) expansion factor for the MBConv block
        dropout: (float) dropout probability; default=0 no dropout effect
        pretrained: (bool) if true, return pretrained model
        progress: (bool) if true, display download progress of pretrained model
        **kwargs: pointers to additional parameters

    Return:
        MobileNet class object
    """
    model = MobileNet(block, layers, expansion=expansion, width_mult=width_mult, dropout=dropout, **kwargs)
    if pretrained:
        if arch in model_urls.keys():
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            model.load_state_dict(state_dict)

    return model

def mobilenet20_1p0_t3(pretrained=False, progress=False, **kwargs):
    """
    mobilenet20
    - width multiplier = 1.0
    - expansion = 3
    - no dropout
    """
    return _mobilenet('mobilenet20', MBConv, [1, 2, 3, 4, 3, 3, 1], width_mult=1.0, expansion=3,
                      dropout=0, pretrained=pretrained, progress=progress, **kwargs)


def mobilenet20_1p0_t4(pretrained=False, progress=False, **kwargs):
    """
    mobilenet20
    - width multiplier = 1.0
    - expansion = 4
    - no dropout
    """
    return _mobilenet('mobilenet20', MBConv, [1, 2, 3, 4, 3, 3, 1], width_mult=1.0, expansion=4,
                      dropout=0, pretrained=pretrained, progress=progress, **kwargs)
