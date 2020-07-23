"""
    MobileNet-V1 from the paper "MobileNets: Efﬁcient Convolutional Neural Networks for
    Mobile Vision Applications" by Andrew G.Howard et al, Google, CoRR 2017

"""

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


# mobilenet variants
__all__ = [
    'MobileNetV1', 'mobilenetv1_28_1p25_32', 'mobilenetv1_28_1p0_32', 'mobilenetv1_28_0p75_32',
    'mobilenetv1_28_0p5_32', 'mobilenetv1_28_0p25_32',
]

# pretrained model urls
model_urls = {

}

# define a 3x3-conv filter preserving input/output fmap dimensions when stride=1
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    3x3 conv filter
    - preserving dimensions when stride=1
    - exactly halve dimensions when stride=2
    - is depthwise separable when groups = in_planes = out_planes
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups,
                     padding=dilation, dilation=dilation, bias=False)

# define a 1x1-conv filter preserving input/output fmap dimensions
def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """
    1x1 conv filter
    - preserving dimensions when stride=1 (by padding=0 and dilation=1; pytorch defaults)
    - exactly halve dimensions when stride=2
    - aka pointwise filter
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups,
                     padding=0, dilation=1, bias=False)

                

class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Conv block

    structure:
    - 3x3-conv-dw s=1/2 -> BN -> relu -> 1x1-conv s=1 -> BN -> relu

    """

    def __init__(self, inplanes, stride=1, norm_layer=None):
        """
        Constructor

        Args:
            inplanes: (int) number of input channels
            stride: (int) stride of first depthwise layer
            norm_layer: (nn.Module) normalization layer; default = BN

        Note:
            - # of output channels = inplanes * stride
        """

        super(DepthwiseSeparableConv2d, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.outplanes = inplanes * stride

        self.convdw1 = conv3x3(inplanes, inplanes, stride=stride, groups=inplanes)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(inplanes, self.outplanes, stride=1)
        self.bn2 = norm_layer(self.outplanes)

        self.dropout = nn.Dropout(p=0.2)

    def _forward_imp1(self, x):
        """ forward method with dropout """

        x = self.relu(self.bn1(self.convdw1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x

    def _forward_imp2(self, x):
        """ forward method without dropout """

        x = self.relu(self.bn1(self.convdw1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

    def forward(self, x):
        """ forward method """
        return self._forward_imp2(x)


class MobileNetV1(nn.Module):
    """
    MobileNetV1

    Structure:

        ImageNet (from original paper; 30 weighted layers):
        - 3x3-conv s=2 > BN > ReLU
        - stack-1 (repeat x3)
            - 3x3-conv-dw s=1 > BN > ReLU > 1x1-conv s=1 > BN > ReLU
            - 3x3-conv-dw s=2 > BN > ReLU > 1x1-conv s=1 > BN > ReLU
            - note: first 1x1-conv filter in the stack has out_planes = 2*in_planes even stride=1
              in preceding dw layer
        - stack-2 (repeat x5)
            - 3x3-conv-dw s=1 > BN > ReLU > 1x1-conv s=1
        - 3x3-conv-dw s=2 > BN > ReLU > 1x1-conv s=1
        - 3x3-conv-dw s=1 > BN > ReLU > 1x1-conv s=1
        - global avg pooling s=1 > FC > softmax

        CIFAR-10/100 (adapted; 24 weighted layers):
        - 3x3-conv s=1 > BN > ReLU
        - stack-1 (repeat x2)
            - 3x3-conv-dw s=1 > BN > ReLU > 1x1-conv s=1 > BN > ReLU
            - 3x3-conv-dw s=2 > BN > ReLU > 1x1-conv s=1 > BN > ReLU
            - note: first 1x1-conv filter in the stack has out_planes = 2*in_planes even stride=1
              in preceding dw layer
        - stack-2 (repeat x5)
            - 3x3-conv-dw s=1 > BN > ReLU > 1x1-conv s=1 > BN > ReLU
        - 3x3-conv-dw s=2 > BN > ReLU > 1x1-conv s=1
        - global avg pooling s=1 > FC > softmax

    """
    def __init__(self, block, layers, width_mult=1.0, res_mult=1.0, norm_layer=None, num_classes=10):
        """
        Constructor

        Args:
            block: (nn.Module) building block of MobileNet; default = DepthwiseSeparableConv2d
            layers: (list) list of two integers for the number of layers per stack
            width_mult: (float) width multiplier; scales # of channels per layer
            res_mult: (float) resolution multiplier; scales input tensor resolutions
            norm_layer: (nn.Module) normalization layer; default = BN
            num_classes: (int) number of classes in dataset
        """
        super(MobileNetV1, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        # set base planes (the number of output channels after first conv filter)
        # on cifar -> 8 (custom); on ImageNet -> 32 (follows original paper)
        self.inplanes = int(8 * width_mult)

        self.conv1 = conv3x3(3, self.inplanes, stride=1)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.convdw2 = conv3x3(self.inplanes, self.inplanes, stride=1, groups=self.inplanes)
        self.bn2 = norm_layer(self.inplanes)
        self.conv3 = conv1x1(self.inplanes, self.inplanes*2, stride=1)
        self.bn3 = norm_layer(self.inplanes*2)

        self.stack1 = self._make_stack_1(block, layers[0], self.inplanes*2)
        self.stack2 = self._make_stack_2(block, layers[1], self.inplanes)
        self.stack3 = self._make_stack_1(block, 1, self.inplanes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, num_classes)

        self.dropout = nn.Dropout(p=0.2)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_stack_1(self, block, num_layers, inplanes, norm_layer=None):
        """
        Stack 1

        structure:
        - 2n layers
            - layer i (i is even) is a DepthwiseSeparableConv block with stride=2
            - layer i+1 is a DepthwiseSeparableConv block with stride=1

        Args:
            block: (nn.Module) block type; default = DepthwiseSeparableConv2d
            num_layers: (int) number of layers in the stack = 2 * num_layers
            inplanes: (int) number of input channels
            norm_layer: (nn.Module) normalization layer; default = self._norm_layer

        note:
        - self.inplanes are updated; after call, self.inplanes = inplanes * (2**num_layers)

        """

        if norm_layer is None:
            norm_layer = self._norm_layer

        layers = []
        self.inplanes = inplanes

        for _ in range(num_layers):
            # append a new layer
            # even layers use stride=2 in 3x3-conv-dw
            layers.append(block(self.inplanes, stride=2, norm_layer=norm_layer))
            # update inplanes
            self.inplanes *= 2
            # odd layers use stride=1 in 3x3-conv-dw
            layers.append(block(self.inplanes, stride=1, norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def _make_stack_2(self, block, num_layers, inplanes, norm_layer=None):
        """
        Stack 2

        structure:
        - layer i is a block with stride=1

        Args:
            block: (nn.Module) default=DepthwiseSeparableConv2d
            num_layers: (int) number of layers in the stack
            inplanes: (int) number of input channels to the stack
            norm_layer: (nn.Module) default = self._norm_layer

        notes:
        - self.inplanes remains constant

        """

        if norm_layer is None:
            norm_layer = self._norm_layer

        layers = []
        self.inplanes = inplanes

        for _ in range(num_layers):
            # append a new layer
            layers.append(block(self.inplanes, stride=1, norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        """ forward method """
                                                                    # batch_size x 3 x 32 x 32
        x = self.relu(self.bn1(self.conv1(x)))                      # batch_size x 8 x 32 x 32
        x = self.relu(self.bn2(self.convdw2(x)))                    # batch_size x 8 x 32 x 32
        x = self.relu(self.bn3(self.conv3(x)))                      # batch_size x 16 x 32 x 32
        x = self.stack1(x)                                          # batch_size x 128 x 4 x 4
        x = self.stack2(x)                                          # batch_size x 128 x 4 x 4
        x = self.stack3(x)                                          # batch_size x 256 x 2 x 2

        x = self.avgpool(x)                                         # batch_size x 256 x 1 x 1
        x = torch.flatten(x, 1)                                     # batch_size x 256*1*1
        out = self.fc(x)                                            # batch_size x 10

        return out


def _mobilenet_v1(arch, block, layers, width_mult, res_mult, pretrained=False, progress=False, **kwargs):
    """
    Abstract generator to build mobilenet-v1

    Args:
        arch: (str) architecture of the pretrained model
        block: (nn.Module) building block of MobileNet; default = DepthwiseSeparableConv2d
        layers: (list) a list of 2 integers representing the number of layers in the two stacks
        width_mult: (float) width multiplier
        res_mult: (float) resolution multiplier
        pretrained: (boolean) if true, load pretrained model if available
        progress: (boolean) if true, display download progress of pretrained model
        **kwargs: pointer to additional parameters

    Return:
        model: (MobileNetV1) returns MobileNet-V1 class object
    """

    model = MobileNetV1(block, layers, width_mult, res_mult, **kwargs)
    if pretrained:
        if arch in model_urls.keys():
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            model.load_state_dict(state_dict)
    return model


def mobilenetv1_28_1p25_32(pretrained=False, progress=True, **kwargs):
    """
    mobilenetv1
    - 28 layers
    - resolution = 32x32 (cifar); res_mult = 1.0
    - width_mult = 1.25
    """
    return _mobilenet_v1('mobilenetv1', DepthwiseSeparableConv2d, [3, 4], 1.25, 1.0, pretrained, progress, **kwargs)

def mobilenetv1_28_1p0_32(pretrained=False, progress=True, **kwargs):
    """
    mobilenetv1
    - 28 layers
    - resolution = 32x32 (cifar); res_mult = 1.0
    - width_mult = 1.0
    """
    return _mobilenet_v1('mobilenetv1', DepthwiseSeparableConv2d, [3, 4], 1.0, 1.0, pretrained, progress, **kwargs)

def mobilenetv1_28_0p75_32(pretrained=False, progress=True, **kwargs):
    """
    mobilenetv1
    - 28 layers
    - resolution = 32x32 (cifar); res_mult = 1.0
    - width_mult = 0.75
    """
    return _mobilenet_v1('mobilenetv1', DepthwiseSeparableConv2d, [3, 4], 0.75, 1.0, pretrained, progress, **kwargs)

def mobilenetv1_28_0p5_32(pretrained=False, progress=True, **kwargs):
    """
    mobilenetv1
    - 28 layers
    - resolution = 32x32 (cifar); res_mult = 1.0
    - width_mult = 0.5
    """
    return _mobilenet_v1('mobilenetv1', DepthwiseSeparableConv2d, [3, 4], 0.5, 1.0, pretrained, progress, **kwargs)

def mobilenetv1_28_0p25_32(pretrained=False, progress=True, **kwargs):
    """
    mobilenetv1
    - 28 layers
    - resolution = 32x32 (cifar); res_mult = 1.0
    - width_mult = 0.25
    """
    return _mobilenet_v1('mobilenetv1', DepthwiseSeparableConv2d, [3, 4], 0.25, 1.0, pretrained, progress, **kwargs)
