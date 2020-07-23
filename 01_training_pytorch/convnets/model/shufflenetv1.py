"""
    ShuffleNetV1 from the paper
    "ShufﬂeNet: An Extremely Efﬁcient Convolutional Neural Network for Mobile Devices"
    by Xiangyu Zhang et al, Face++, CVPR 2018

"""

# imports
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# shufflenetv1 variants
__all__ = [
    'ShuffleNetV1',
    'shufflenetv1_50_s1p0_g1', 'shufflenetv1_50_s1p0_g2', 'shufflenetv1_50_s1p0_g3',
    'shufflenetv1_50_s1p0_g4', 'shufflenetv1_50_s1p0_g8', 'shufflenetv1_50_s0p5_g1',
    'shufflenetv1_50_s0p5_g2', 'shufflenetv1_50_s0p5_g3', 'shufflenetv1_50_s0p5_g4',
    'shufflenetv1_50_s0p5_g8'
]


# pretrained models
model_urls = {

}


# 3x3 conv filter preserving fmap dimensions
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    3x3 conv filter
    - preserves input/output dimensions if stride=1
    - exactly halves output dimensions if stride=2
    - is depthwise if in_planes = out_planes = groups
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups,
                     padding=dilation, dilation=dilation, bias=False)

# 1x1 conv filter preserving fmap dimensions
def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """
    1x1 conv filter
    - preserves input/output dimensions if stride=1
    - exactly halves output dimensions if stride=2
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups,
                     padding=0, dilation=1, bias=False)


class ShuffleUnitV1(nn.Module):
    """
    Basic unit for ShuffleNetV1

    structure:
    - 1x1-Gconv > bn > relu > channel shuffle > 3x3-conv-dw s=1/2 > bn > 1x1-Gconv > bn > skip add/concat > relu

    notes:
    - if intermediate 3x3-conv-dw layer has stride=2, skip connection goes through downsampling by 3x3-avgpool w.t. stride=2
        - also use concat instead of add in this case (to increase output channel numbers)
    - block output channels = inplanes * stride always holds for stride = 1 or 2
    - bottleneck channels = inplanes / expansion = groups * base_width
        - note that this always holds even for stride > 1, b.c. in this case final block output is doubled by concat
          with the skip path, but the bottleneck path still has output channels = inplanes
    """

    expansion = 4

    def __init__(self, inplanes, groups=1, group_factor=None, base_width=18, stride=1,
                 scale_factor=1, dropout=0, norm_layer=None, downsample=None, first_block=False):
        """
        Constructor

        Args:
            inplanes: (int) number of input channels to the block
            groups: (int) number of groups used by the grouped conv filters in the block
            group_factor: (int) width = group_factor * base_width; default = groups
            base_width: (int) number of channels per group
            stride: (int) stride; applied to middle 3x3-conv-dw layer
            scale_factor: (float) equivalent to width multiplier; network width scaled by scale_factor
            dropout: (float) p = dropout; if = 0 no dropout effect
            norm_layer: (nn.Module) normalization layer; default = nn.BatchNorm2d
            downsample: (nn.Sequential) downsamples skip connection by 3x3-avgpool if stride > 1
            first_block: (bool) if true, deduct 6 from bottleneck width (according to original paper)

        """
        super(ShuffleUnitV1, self).__init__()

        self._base_width = base_width
        self._groups = groups

        if group_factor is None:
            group_factor = self._groups

        self._width = int(self._base_width * group_factor)

        if first_block:
            # accoring to original paper, requires bottleneck width to deduct 6 in first block in stage 1
            # e.g., for g=1, at stack1 first block, 144 = 24 + 120 -> width = 120 / 4 = 30
            # but for other blocks and other stacks, width = 144 / 4 = 36 = 30 + 6
            # this is the case for all groups, because groups would change, it is best to deduct 6 here
            self._width -= int(6 * scale_factor)

        self._outplanes = int(self._width * self.expansion)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.gconv1 = conv1x1(inplanes, self._width, stride=1, groups=groups)
        self.bn1 = norm_layer(self._width)
        self.convdw2 = conv3x3(self._width, self._width, stride=stride, groups=self._width)
        self.bn2 = norm_layer(self._width)
        self.gconv3 = conv1x1(self._width, self._outplanes, stride=1, groups=groups)
        self.bn3 = norm_layer(self._outplanes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.downsample = downsample


    def _channel_shuffle(self, x):
        """
        channel shuffle operator

        - input tensor = batch_size x groups*base_width x H x H
        - 1): split tensor along channel axis by groups
        - 2): stack the list of splitted tensors along a new dimension (dim=1) for the groups axis
            - resulting tensor would have axes of batch, groups, channels, H, H
        - 3): transpose tensor to swap groups and channels axes
        - 4): flatten the tensor back from groups axis to channels axis
            - resulting tensor would be a channel shuffled version of input tensor with same shape

        notes:
        - does not contain weighted layers, so no need to declare in self.__init__()
        """
        # split
        x_lst = torch.split(x, self._groups, dim=1)
        # stack
        y = torch.stack(x_lst, dim=1)
        # transpose
        y = torch.transpose(y, dim0=1, dim1=2)
        # flatten
        y = torch.flatten(y, start_dim=1, end_dim=2)
        # check tensor shapes
        assert x.shape == y.shape

        return y


    def forward(self, x):
        """ forward method """

        identity = x

        out = self.bn1(self.gconv1(x))
        out = self.dropout(out)
        out = self.relu(out)
        # shuffle channels after first grouped convolution layer
        out = self._channel_shuffle(out)
        out = self.bn2(self.convdw2(out))
        out = self.dropout(out)
        out = self.bn3(self.gconv3(out))
        out = self.dropout(out)

        # if stride=2, downsamples skip connection then concat along channel dimension
        if self.downsample is not None:
            identity = self.downsample(x)
            # by concat along channel dimension, doubles number of channels when stride=2
            out = torch.cat((out, identity), dim=1)

        out = self.relu(out)

        return out



class ShuffleNetV1(nn.Module):
    """
    ShuffleNetV1

    Structure:

    > ImageNet (from original paper)
    - 3x3-conv s=2 > bn > relu > 3x3-maxplool s=2
    - stack1 x4
    - stack2 x8
    - stack3 x4
    - global avgpool > fc

    > Cifar-10/100 (adapted)
    - 3x3-conv s=1 > bn > relu
    - stack1 x4
    - stack2 x8
    - stack3 x4
    - global avgpool > fc

    notes:
    - first block in each stack has stride=2, rest of the blocks all have stride=1
    - stack output channels = stack input channels * stride
        - output channels of each layer of block in the stack is always equal to stack output channels
        - for shufflenetv1, block outplanes = groups * base_width
            - groups is a fixed number for a network variant
            - hence must set stack base_width = base_width * stride to match progression of outplanes
    """

    def __init__(self, block, layers, groups=1, base_width=36, scale_factor=1,
                 num_classes=10, dropout=0, norm_layer=None):
        """
        Constructor

        Args
            block: (nn.Module) ShuffleUnitV1 or ShuffleUnitV2
            layers: (list of ints) a list of 3 integers for the number of blocks per stack
            groups: (int) number of groups used by the grouped conv filters in the block
            base_width: (int) number of channels per group
            scale_factor: (float) equivalent to width multiplier; network width scaled by scale_factor
            num_classes: (int) number of classes in the dataset
            dropout: (float) if = 0 disables dropout effect
            norm_layer: (nn.Module) normalization layer; default = nn.BatchNorm2d
        """

        super(ShuffleNetV1, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer

        self._scale_factor = scale_factor
        self._inplanes = int(24 * self._scale_factor)
        self._base_width = int(base_width * self._scale_factor)
        self._groups = groups
        self._dropout = dropout
        self._expansion = 4

        self.conv1 = conv3x3(3, self._inplanes, stride=1)
        self.bn1 = norm_layer(self._inplanes)

        # first block as a stand-alone stack, uses groups=1 (no grouped conv)
        self.stack2 = self._make_stack(block, 1, self._inplanes, stride=2, groups=1, first_block=True)
        self.stack3 = self._make_stack(block, layers[0], self._inplanes, stride=1)
        self.stack4 = self._make_stack(block, layers[1], self._inplanes, stride=2)
        self.stack5 = self._make_stack(block, layers[2], self._inplanes, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self._inplanes, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_stack(self, block, num_layers, inplanes, stride=1, groups=None, first_block=False):
        """
        build shufflenet stack

        Args:
            block: (nn.Module) ShuffleUnitV1 or ShuffleUnitV2
            num_layers: (int) number of layers in the stack
            inplanes: (int) number of input channels to the stack
            stride: (int) stride applied to 3x3-conv-dw layer
            groups: (int) number of groups applied to grouped conv filters; default = self._groups
            first_block: (bool) if true, deduct 6 from self._width in first block of first stage

        notes:
        - number of outplanes of each block in the stack always = inplanes * stride

        """

        norm_layer = self._norm_layer
        downsample = None
        group_factor = None
        scale_factor = 1

        if groups is None:
            groups = self._groups

        # if stride != 1, apply downsample on identity shortcut
        if stride != 1:
            downsample = nn.Sequential(
                # exactly halves fmap dimensions if padding=1, 3x3 kernel, stride=2
                nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
            )

        # set group_factor to be self._groups for first block in first stack
        # because this block always uses groups=1 for convolution, but the actual group_factor should not be fixed to 1
        if first_block:
            group_factor = self._groups
            scale_factor = self._scale_factor

        layers = []

        layers.append(block(inplanes, groups=groups, group_factor=group_factor, base_width=self._base_width,
                            stride=stride, scale_factor=scale_factor, dropout=self._dropout,
                            norm_layer=norm_layer, downsample=downsample, first_block=first_block))

        # block base_width needs to be updated for current stack except for first block in first stack
        if not first_block:
            self._base_width *= stride

        # regardless of block type, block output channels = input channels * stride always holds
        self._inplanes = self._base_width * self._groups * self._expansion

        for _ in range(1, num_layers):
            layers.append(block(self._inplanes, groups=groups, base_width=self._base_width, stride=1,
                                dropout=self._dropout, norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        """ forward method """

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)
        x = self.stack5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def _shufflenet(arch, block, layers, groups=1, base_width=18, scale_factor=1, dropout=0,
                pretrained=False, progress=False, **kwargs):
    """
    Abstract generator to build shufflenet variants

    Args:
        arch: (str) architecture of pretrained model
        block: (nn.Module) block type
        layers: (list of ints) list of integers representing the number of layers/blocks per stack
        groups: (int) number of groups used throughout the network for grouped convolution
        base_width: (int) parameter for ShuffleUnitV1, associated with groups
        scale_factor: (float) equivalent to width multiplier
        dropout: (float) controls dropout behavior; default no dropout
        pretrained: (bool) if true download pretrained models from url
        progress: (bool) if true display download progress
        **kwargs: pointer to additional parameters
    """
    model = ShuffleNetV1(block, layers, groups, base_width, scale_factor, dropout=dropout, **kwargs)
    if pretrained:
        if arch in model_urls.keys():
            state_dict = load_state_dict_from_url(model[arch], progress=progress)
            model.load_state_dict(state_dict)
    return model

def shufflenetv1_50_s1p0_g1(dropout=0, pretrained=False, progress=False, **kwargs):
    """
    shufflenetv1
    - layers = 50
    - scaling factor = 1.0
    - group factor = 1
    """
    return _shufflenet('shufflenetv1', ShuffleUnitV1, [3, 8, 4], groups=1, base_width=36, scale_factor=1,
                       dropout=dropout, pretrained=pretrained, progress=progress, **kwargs)

def shufflenetv1_50_s1p0_g2(dropout=0, pretrained=False, progress=False, **kwargs):
    """
    shufflenetv1
    - layers = 50
    - scaling factor = 1.0
    - group factor = 2
    """
    return _shufflenet('shufflenetv1', ShuffleUnitV1, [3, 8, 4], groups=2, base_width=25, scale_factor=1,
                       dropout=dropout, pretrained=pretrained, progress=progress, **kwargs)


def shufflenetv1_50_s1p0_g3(dropout=0, pretrained=False, progress=False, **kwargs):
    """
    shufflenetv1
    - layers = 50
    - scaling factor = 1.0
    - group factor = 3
    """
    return _shufflenet('shufflenetv1', ShuffleUnitV1, [3, 8, 4], groups=3, base_width=20, scale_factor=1,
                       dropout=dropout, pretrained=pretrained, progress=progress, **kwargs)


def shufflenetv1_50_s1p0_g4(dropout=0, pretrained=False, progress=False, **kwargs):
    """
    shufflenetv1
    - layers = 50
    - scaling factor = 1.0
    - group factor = 4
    """
    return _shufflenet('shufflenetv1', ShuffleUnitV1, [3, 8, 4], groups=4, base_width=17, scale_factor=1,
                       dropout=dropout, pretrained=pretrained, progress=progress, **kwargs)


def shufflenetv1_50_s1p0_g8(dropout=0, pretrained=False, progress=False, **kwargs):
    """
    shufflenetv1
    - layers = 50
    - scaling factor = 1.0
    - group factor = 8
    """
    return _shufflenet('shufflenetv1', ShuffleUnitV1, [3, 8, 4], groups=8, base_width=12, scale_factor=1,
                       dropout=dropout, pretrained=pretrained, progress=progress, **kwargs)


def shufflenetv1_50_s0p5_g1(dropout=0, pretrained=False, progress=False, **kwargs):
    """
    shufflenetv1
    - layers = 50
    - scaling factor = 0.5
    - group factor = 1
    """
    return _shufflenet('shufflenetv1', ShuffleUnitV1, [3, 8, 4], groups=1, base_width=36, scale_factor=0.5,
                       dropout=dropout, pretrained=pretrained, progress=progress, **kwargs)


def shufflenetv1_50_s0p5_g2(dropout=0, pretrained=False, progress=False, **kwargs):
    """
    shufflenetv1
    - layers = 50
    - scaling factor = 0.5
    - group factor = 2
    """
    return _shufflenet('shufflenetv1', ShuffleUnitV1, [3, 8, 4], groups=2, base_width=25, scale_factor=0.5,
                       dropout=dropout, pretrained=pretrained, progress=progress, **kwargs)


def shufflenetv1_50_s0p5_g3(dropout=0, pretrained=False, progress=False, **kwargs):
    """
    shufflenetv1
    - layers = 50
    - scaling factor = 0.5
    - group factor = 3
    """
    return _shufflenet('shufflenetv1', ShuffleUnitV1, [3, 8, 4], groups=3, base_width=20, scale_factor=0.5,
                       dropout=dropout, pretrained=pretrained, progress=progress, **kwargs)


def shufflenetv1_50_s0p5_g4(dropout=0, pretrained=False, progress=False, **kwargs):
    """
    shufflenetv1
    - layers = 50
    - scaling factor = 0.5
    - group factor = 4
    """
    return _shufflenet('shufflenetv1', ShuffleUnitV1, [3, 8, 4], groups=4, base_width=17, scale_factor=0.5,
                       dropout=dropout, pretrained=pretrained, progress=progress, **kwargs)


def shufflenetv1_50_s0p5_g8(dropout=0, pretrained=False, progress=False, **kwargs):
    """
    shufflenetv1
    - layers = 50
    - scaling factor = 0.5
    - group factor = 8
    """
    return _shufflenet('shufflenetv1', ShuffleUnitV1, [3, 8, 4], groups=8, base_width=12, scale_factor=0.5,
                       dropout=dropout, pretrained=pretrained, progress=progress, **kwargs)
