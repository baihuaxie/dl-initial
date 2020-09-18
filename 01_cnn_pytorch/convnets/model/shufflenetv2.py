"""
    ShuffleNetV2 fromthe paper
    "ShuﬄeNet V2: Practical Guidelines for Eﬃcient CNN Architecture Design"
    by Ningning Ma et al, Face++, EECV 2018

"""

# imports
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# shufflenetv1 variants
__all__ = [
    'ShuffleNetV2', 'shufflenetv2_51_s0p5', 'shufflenetv2_51_s1p0',
    'shufflenetv2_51_s1p5', 'shufflenetv2_51_s2p0'
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


class ShuffleUnitV2(nn.Module):
    """
    Basic unit for ShuffleNetV2

    structure:
    > stride=1
    - channel split operator > 1x1-conv > bn > relu > 3x3-conv-dw > bn > 1x1-conv > bn > relu > skip connect concat > channel shuffle operator
    > stride=2
    - bottleneck path: 1x1-conv > bn > relu > 3x3-conv-dw s=2 > bn > 1x1-conv > bn > relu > concat > channel shuffle operator
    - skip path: 3x3-conv-dw s=2 > bn > 1x1-conv > bn > relu

    notes:
    - block output channels = block input channels * stride always holds for stride = 1 or 2
    - bottlenck path inplanes = outplanes always holds regardless of stride
        - when stride=2 final block outplanes is doubled but by the concat operation
    - in shufflenetv2 the bottleneck path has expansion=1, so each of the three layers has inplanes = outplanes always
        - authors claim to benefit memory access reduction in this way
    """

    def __init__(self, inplanes, channel_split=0.5, stride=1, dropout=0,
                 norm_layer=None, downsample=None):
        """
        Constructor

        Args:
            inplanes: (int) number of input channels to the block
            channel_split: (float) channel split factor; number of channels going through the bottleneck path = channel_split * inplanes
            stride: (int) stride; applied to 3x3-conv-dw if stride > 1
            dropout: (float) p=dropout; if dropout=0 no dropout effect
            norm_layer: (nn.Module) normalization layer; default = nn.BatchNorm2d
            downsample: (nn.Sequential) downsamples the skip path if stride > 1

        """
        super(ShuffleUnitV2, self).__init__()

        self._channel_split_factor = channel_split
        if downsample is None:
            self._width = int(inplanes * self._channel_split_factor)
        else:
            self._width = (inplanes)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # set groups number for channel shuffle = 2 (because channels are splitted into two paths?)
        self._groups = 2

        self.conv1 = conv1x1(self._width, self._width, stride=1)
        self.bn1 = norm_layer(self._width)
        self.convdw2 = conv3x3(self._width, self._width, stride=stride, groups=self._width)
        self.bn2 = norm_layer(self._width)
        self.conv3 = conv1x1(self._width, self._width, stride=1)
        self.bn3 = norm_layer(self._width)

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


    def _channel_split(self, x):
        """
        channel split operator

        - split x into x1 and x2
        - x1 = batch_size x inplanes*channel_split x H x H and goes through the bottleneck path
        - x2 = batch_size x inplanes*(1-channel_split) x H x H and goes through the identity path

        returns a list of two tensors [x1, x2]
        """
        split1 = int(x.shape[1] * self._channel_split_factor)
        split2 = int(x.shape[1] - split1)
        return torch.split(x, [split1, split2], dim=1)



    def forward(self, x):
        """ forward method """

        if self.downsample is None:
            # if stride=1, no downsample, split x into
            # out for bottleneck path + identity for identity path
            # by channel_split
            out, identity = self._channel_split(x)
        else:
            # if stride=2, no channel split
            out = x
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(out)))
        out = self.dropout(out)
        out = self.bn2(self.convdw2(out))
        out = self.dropout(out)
        out = self.relu(self.bn3(self.conv3(out)))

        out = torch.cat([out, identity], dim=1)

        out = self._channel_shuffle(out)

        return out


class ShuffleNetV2(nn.Module):
    """
    ShuffleNetV2

    Structure:

    > ImageNet (from original paper)
    - 3x3-conv s=2 > bn > relu > 3x3-maxplool s=2
    - stack1 x4
    - stack2 x8
    - stack3 x4
    - 1x1-conv > bn > relu > global avgpool > fc

    > Cifar-10/100 (adapted)
    - 3x3-conv s=1 > bn > relu
    - stack1 x4
    - stack2 x8
    - stack3 x4
    - 1x1-conv > bn > relu > global avgpool > fc

    notes:
    - first block in each stack has stride=2, rest of the blocks all have stride=1
    - stack output channels = stack input channels * stride
        - output channels of each layer of block in the stack is always equal to stack output channels
    - in original paper it is difficult to understand the scaling factor, so here I've adapted width = 48 * scale_factor

    """

    def __init__(self, block, layers, channel_split=1, scale_factor=1, num_classes=10,
                 dropout=0, norm_layer=None):
        """
        Constructor

        Args
            block: (nn.Module) ShuffleUnitV1 or ShuffleUnitV2
            layers: (list of ints) a list of 3 integers for the number of blocks per stack
            channel_split: (float) channel split factor; number of channels going through the bottleneck path = channel_split * inplanes
            scale_factor: (float) equivalent to width multiplier; network width scaled by scale_factor
            num_classes: (int) number of classes in the dataset
            dropout: (float) if = 0 disables dropout effect
            norm_layer: (nn.Module) normalization layer; default = nn.BatchNorm2d
        """

        super(ShuffleNetV2, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer

        self._inplanes = int(48 * scale_factor)
        self._channel_split = channel_split
        self._dropout = dropout

        self.conv1 = conv3x3(3, 24, stride=1)
        self.bn1 = norm_layer(24)

        self.stack2 = self._make_stack(block, layers[0], 24, stride=2)
        self.stack3 = self._make_stack(block, layers[1], self._inplanes, stride=2)
        self.stack4 = self._make_stack(block, layers[2], self._inplanes, stride=2)

        self.conv5 = conv1x1(self._inplanes, 1024, stride=1)
        self.bn5 = norm_layer(1024)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

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

    def _make_stack(self, block, num_layers, inplanes, stride=1):
        """
        build shufflenet stack

        Args:
            block: (nn.Module) ShuffleUnitV1 or ShuffleUnitV2
            num_layers: (int) number of layers in the stack
            inplanes: (int) number of input channels to the stack
            stride: (int) stride applied to 3x3-conv-dw layer

        notes:
        - number of outplanes of each block in the stack always = inplanes * stride

        """

        norm_layer = self._norm_layer
        downsample = None

        # if stride != 1, apply downsample on identity shortcut
        if stride != 1:
            downsample = nn.Sequential(
                # depthwise conv
                conv3x3(inplanes, inplanes, stride=2, groups=inplanes),
                norm_layer(inplanes),
                conv1x1(inplanes, inplanes, stride=1),
                norm_layer(inplanes),
                nn.ReLU(inplace=True)
            )

        layers = []

        layers.append(block(inplanes, channel_split=self._channel_split, stride=stride,
                            dropout=self._dropout, norm_layer=norm_layer, downsample=downsample))

        # block output channels = input channels * stride always holds
        self._inplanes = inplanes * stride

        for _ in range(1, num_layers):
            layers.append(block(self._inplanes, channel_split=self._channel_split, stride=1,
                                dropout=self._dropout, norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        """ forward method """

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)
        x = self.relu(self.bn5(self.conv5(x)))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


def _shufflenet(arch, block, layers, channel_split=1, scale_factor=1, dropout=0,
                pretrained=False, progress=False, **kwargs):
    """
    Abstract generator to build shufflenet variants

    Args:
        arch: (str) architecture of pretrained model
        block: (nn.Module) block type
        layers: (list of ints) list of integers representing the number of layers/blocks per stack
        channel_split: (float) channel split factor
        scale_factor: (float) equivalent to width multiplier
        dropout: (float) controls dropout behavior; default no dropout
        pretrained: (bool) if true download pretrained models from url
        progress: (bool) if true display download progress
        **kwargs: pointer to additional parameters
    """
    model = ShuffleNetV2(block, layers, channel_split, scale_factor, dropout=dropout, **kwargs)
    if pretrained:
        if arch in model_urls.keys():
            state_dict = load_state_dict_from_url(model[arch], progress=progress)
            model.load_state_dict(state_dict)
    return model


def shufflenetv2_51_s1p0(dropout=0, pretrained=False, progress=False, **kwargs):
    """
    shufflenetv2
    - layers = 51
    - scaling factor = 1.0
    """
    return _shufflenet('shufflenetv1', ShuffleUnitV2, [3, 8, 4], channel_split=0.5, scale_factor=1,
                       dropout=dropout, pretrained=pretrained, progress=progress, **kwargs)


def shufflenetv2_51_s0p5(dropout=0, pretrained=False, progress=False, **kwargs):
    """
    shufflenetv2
    - layers = 51
    - scaling factor = 0.5
    """
    return _shufflenet('shufflenetv1', ShuffleUnitV2, [3, 8, 4], channel_split=0.5, scale_factor=0.5,
                       dropout=dropout, pretrained=pretrained, progress=progress, **kwargs)


def shufflenetv2_51_s1p5(dropout=0, pretrained=False, progress=False, **kwargs):
    """
    shufflenetv2
    - layers = 51
    - scaling factor = 1.5
    """
    return _shufflenet('shufflenetv1', ShuffleUnitV2, [3, 8, 4], channel_split=0.5, scale_factor=1.5,
                       dropout=dropout, pretrained=pretrained, progress=progress, **kwargs)


def shufflenetv2_51_s2p0(dropout=0, pretrained=False, progress=False, **kwargs):
    """
    shufflenetv2
    - layers = 51
    - scaling factor = 2.0
    """
    return _shufflenet('shufflenetv1', ShuffleUnitV2, [3, 8, 4], channel_split=0.5, scale_factor=2,
                       dropout=dropout, pretrained=pretrained, progress=progress, **kwargs)
