"""
DenseNet from the paper "Densely Connected Convolutional Networks", Gao Huang et al, CVPR 2017

"""

import torch
import torch.nn as nn


# densenet variants
__all__ = [
    'DenseNet', 'densenet40_k12', 'densenet100_k12', 'densenet100_k24',
    'densenetbc100_k12', 'densenetbc250_k24', 'densenetbc190_k40'
]

def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=0, dilation=1):
    """
    define a 3x3 conv-2d with padding=dilation=1

    purpose:
        - preserves in/out fmap dimensions when stride=1
        - let output dimension = exactly half of input dimension when stride=2

    this works for 3x3 kernel sizes only (the most commonly used convolutional kernels)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups,
                     padding=dilation, dilation=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """
    define a 1x1 conv-2d with padding=0, dilation=1 (defaults)

    purpose:
        - preserve in/out fmap dimensions when stride=1
        - halve output fmap dimensions exactly when stride=2

    this works for 1x1 kernel sizes (commonly used as bottleneck layers in conv-nets)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DenseLayer(nn.Module):
    """
    Composite layer for Dense Block (vanilla DenseNet)

    structure: BN -> relu -> conv-3x3

    """

    def __init__(self, inplanes: int, outplanes: int, norm_layer: nn.Module = None):
        """
        Constructor

        Args:
            inplanes: (int) number of input channels
            outplanes: (int) number of output channels
            norm_layer: (nn.Module) normalization layer; default is BatchNorm2d
        """

        super(DenseLayer, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, outplanes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ forward method"""
                                                                            # x = batch_size x inplanes x H x H
        out = self.conv1(self.relu(self.bn1(x)))                            # out = batch_size x outplanes x H x H
        return out


class BottleneckLayer(nn.Module):
    """
    Bottleneck + Composite layer for Dense Block (DenseNet-B)

    structure: BN -> relu -> conv-1x1 -> BN -> relu -> conv-3x3

    """

    expansion = 4

    def __init__(self, inplanes: int, outplanes: int, norm_layer: nn.Module = None):
        """
        Constructor

        Args:
            inplanes: (int) number of input channels
            outplanes: (int) number of output channels
            norm_layer: (nn.Module) normalization layer; default=BatchNorm2d

        """
        super(BottleneckLayer, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.bn1 = norm_layer(inplanes)
        self.conv1 = conv1x1(inplanes, outplanes*self.expansion)
        self.bn2 = norm_layer(outplanes*self.expansion, outplanes*self.expansion)
        self.conv2 = conv3x3(outplanes*self.expansion, outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ forward method """
                                                                            # x = batch_size x inplanes x H x H
        x = self.relu(self.bn1(x))                                          # x = batch_size x inplanes x H x H
        x = self.conv1(x)                                                   # x = batch_size x outplanes*expansion x H x H
        x = self.relu((self.bn2(x)))                                        # x = batch_size x outplanes*expansion x H x H
        x = self.conv2(x)                                                   # x = batch_size x outplanes x H x H

        return x


class TransitionLayer(nn.Module):
    """
    Transition layer with option of compression
        - performs downsampling in between consecutive Dense layers
        - allows the choice to reduce number of channels in compression mode by setting theta < 1

    structure: conv-1x1 -> stride=2 average pooling 2x2

    """

    def __init__(self, inplanes: int, stride: int = 2, theta: float = 1.0):
        """
        Constructor:

        Args:
            inplanes: (int) number of input channels; number of output channels = inplanes * theta
            stride: (int) striding
            theta: (float) the compression factor; 0 < theta < 1 (when theta = 0.5 referred to as DenseNet-C)
        """

        super(TransitionLayer, self).__init__()

        self.conv1 = conv1x1(inplanes, int(inplanes * theta))
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=stride, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ forward method """
                                                                            # x = batch_size x inplanes x H x H
        x = self.avgpool(self.conv1(x))                                     # x = batch_size x inplanes*theta x H/2 x H/2
        return x


class DenseNet(nn.Module):
    """
    DenseNet

    Structure:

        CIFAR-10/100:
        - conv-3x3 (no stride) -> bn -> relu
        - dense-block 1 (bn -> relu -> conv-3x3; or bn -> relu -> conv-1x1 -> bn -> relu -> conv-3x3)
        - transition layer 1 (bn -> relu -> conv-1x1 -> stride=2 average pooling)
        - dense-block 2
        - transition layer 2
        - dense-block 3
        - global average pooling -> 10d/100d fc -> softmax

        ImageNet:
        - conv-7x7 stride=2 -> bn -> relu -> max-pooling-3x3 stride=2
        - dense-block 1
        - transition layer 1
        - dense-block 2
        - transition layer 2
        - dense-block 3
        - transition layer 3
        - dense-block 4
        - global average pooling -> 1000d fc -> softmax
    """

    def __init__(self, layer, layers, growth_factor=12, compression_factor=1, num_classes=10, norm_layer=None):
        """
        Constructor

        Args:
            layer: (DenseLayer or BottleneckLayer obj) type of dense layer used to build the dense layers
            layers: (list) a list of 3 integers, each specifying the number of dense layers in each of the 3 dense layers
            num_classes: (int) the number of classes to predict; also the output of final linear layer
            growth_factor: (int) a hyperparameter specifying the number of output channels for each dense block in the network
            compression_factor: (float) a hyperparameter (denoted by theta) specifying the number of output channels in transition layers being theta*input_channels
            norm_layer: (nn.Module) normalization layer; default=BatchNorm2d

        """

        super(DenseNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.k = growth_factor
        self.theta = compression_factor
        self._layer = layer
        self._layers = layers

        self._inplanes = []
        self._outplanes = []
        self._inplanes.append(self.k * 2)

        # first layer is a convolution with no striding, outplanes = 2*growth_factor, inplanes = 3
        # here I chose to use 3x3 convolution with padding=1, because fmap sizes are 32x32, maybe no need to use 7x7 filters
        # followed by bn and relu, no pooling
        self.conv1 = conv3x3(3, self._inplanes[0])
        self.bn1 = norm_layer(self._inplanes[0])
        self.relu = nn.ReLU(inplace=True)

        # calculate dense block input and output channels
        # on CIFAR-10/100 dataset, DenseNet has 3 dense blocks, each having number of layers = layers[i], where i = 0, 1, 2
        # for each dense block, the number of output channels of each layer is always equal to the growth_factor
        # the number of input channels of layer L in a dense block is calculated as = k0 + k*(L-1); where k is the growth_factor, k0 is the
        # number of input channels to the first layer
        for i in range(len(layers)):
            # the output of each dense block is = growth_factor * number of layers + input_channels
            self._outplanes.append(self.k * layers[i] + self._inplanes[i])
            # the input of each dense block except the first one is always = output of transition layer = output of last dense block * compression_factor
            self._inplanes.append(int(self._outplanes[i] * self.theta))

        # dense + transition layers
        self.denseblock1 = self._init_denselayers(layer, self._inplanes[0], layers[0])
        self.transition1 = TransitionLayer(self._outplanes[0], stride=2, theta=self.theta)
        self.denseblock2 = self._init_denselayers(layer, self._inplanes[1], layers[1])
        self.transition2 = TransitionLayer(self._outplanes[1], stride=2, theta=self.theta)
        self.denseblock3 = self._init_denselayers(layer, self._inplanes[2], layers[2])

        # average pooling + linear + softmax layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self._outplanes[2], num_classes)

        self.dropout = nn.Dropout(p=0.2)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _init_denselayers(self, layer, inplanes, num_layers, norm_layer=None):
        """
        Define the dense layers in a dense block

        Args:
            layer: (DenseLayer or BottleneckLayar object) type of dense layer in the dense block
            inplanes: (int) number of input channels to the dense block
            num_layers: (int) number of dense layers in the dense block
            norm_layer: (nn.Module) normalization layer; default = self._norm_layer

        Returns:
            denseblock: (nn.ModuleList) a list containing the layers in the denseblock, but not connected

        """
        if norm_layer is None:
            norm_layer = self._norm_layer

        layers = []

        for i in range(num_layers):
            layer_input = self.k * i + inplanes
            layer_output = self.k

            layers.append(layer(layer_input, layer_output, norm_layer))

        return nn.ModuleList(layers)


    def _make_denseblock(self, x: torch.Tensor, block):
        """
        Constructing a DenseBlock

        Args:
            x: (torch.Tensor) input tensor
            block: (nn.ModuleList) a list containing the layers defined in the denseblock

        Return:
            x: (torch.Tensor) output tensor
        """

        for i in range(len(block)):
            layer_input = block[i].conv1.in_channels
            if layer_input != x.size()[1]:
                raise ValueError("input tensor dimension {} does not match with input channel number {} for layer {} in denseblock".format(
                                 x.size()[1], layer_input, i))
            x = torch.cat((x, block[i](x)), dim=1)

        return x


    def _forward_imp1(self, x: torch.Tensor):
        """ forward method: no dropout """
                                                                                                # eg: k = 12, layers = [32, 32, 32], theta=0.5
                                                                                                # x = batch_size x 3 x 32 x 32
        x = self.relu(self.bn1(self.conv1(x)))                                                  # x = batch_size x 24 x 32 x 32
        x = self._make_denseblock(x, self.denseblock1)                                          # x = batch_size x 408 x 32 x 32 (408 = 24 + 12*32)
        x = self.transition1(x)                                                                 # x = batch_size x 204 x 16 x 16 (204 = 408*0.5)
        x = self._make_denseblock(x, self.denseblock2)                                          # x = batch_size x 588 x 16 x 16 (588 = 204 + 12*32)
        x = self.transition2(x)                                                                 # x = batch_size x 294 x 8 x 8   (294 = 588*0.5)
        x = self._make_denseblock(x, self.denseblock3)                                          # x = batch_size x 678 x 8 x 8   (678 = 294 + 12*32)
        x = self.avgpool(x)                                                                     # x = batch_size x 678 x 1 x 1
        x = torch.flatten(x, 1)                                                                 # x = batch_size x 678
        out = self.fc(x)                                                                        # x = batch_size x num_classes

        return out


    def _forward_imp2(self, x: torch.Tensor):
        """ forward method: dropout after every block """
                                                                                                # eg: k = 12, layers = [32, 32, 32], theta=0.5
                                                                                                # x = batch_size x 3 x 32 x 32
        x = self.relu(self.bn1(self.conv1(x)))                                                  # x = batch_size x 24 x 32 x 32
        x = self._make_denseblock(x, self.denseblock1)                                          # x = batch_size x 408 x 32 x 32 (408 = 24 + 12*32)
        x = self.dropout(x)
        x = self.transition1(x)                                                                 # x = batch_size x 204 x 16 x 16 (204 = 408*0.5)
        x = self.dropout(x)
        x = self._make_denseblock(x, self.denseblock2)                                          # x = batch_size x 588 x 16 x 16 (588 = 204 + 12*32)
        x = self.dropout(x)
        x = self.transition2(x)                                                                 # x = batch_size x 294 x 8 x 8   (294 = 588*0.5)
        x = self.dropout(x)
        x = self._make_denseblock(x, self.denseblock3)                                          # x = batch_size x 678 x 8 x 8   (678 = 294 + 12*32)
        x = self.dropout(x)
        x = self.avgpool(x)                                                                     # x = batch_size x 678 x 1 x 1
        x = torch.flatten(x, 1)                                                                 # x = batch_size x 678
        out = self.fc(x)                                                                        # x = batch_size x num_classes

        return out


    def forward(self, x: torch.Tensor):
        """ forward method """
        return self._forward_imp2(x)




def _densenet(layer, layers, growth, compression, pretrained, progress, arch=None, **kwargs):
    """
    Abstract generator to build a DenseNet

    Args:
        layer: (DenseLayer or BottleneckLayer class obj) layer type used for DenseBlock
        layers: (list) a list of 3 integers each specifying the number of layers in each of the 3 Dense layers (for cifar dataset)
        growth: (int) hyperparameter for growth factor k
        compression: (float) hyperparamter for compression factor theta
        pretrained: (boolean) if True, use pretrained models with weights from pytorch
        progress: (boolean) if True, display a progress bar of the download (of pretrained models) to stderr
        arch: (str) architecture of the pretrained model
        **kwargs: (pointer) to additional parameters

    Return:
        model: (DenseNet class obj) a DenseNet model instance
    """
    model = DenseNet(layer, layers, growth, compression, **kwargs)
    if pretrained:
        if arch is not None:
            pass
    return model


def densenet40_k12(pretrained=False, progress=False, **kwargs):
    """
    DenseNet-40 with k=12
    """
    return _densenet(DenseLayer, [12, 12, 12], 12, 1, pretrained, progress, **kwargs)


def densenet100_k12(pretrained=False, progress=False, **kwargs):
    """
    DenseNet-100 with k=12
    """
    return _densenet(DenseLayer, [32, 32, 32], 12, 1, pretrained, progress, **kwargs)


def densenet100_k24(pretrained=False, progress=False, **kwargs):
    """
    DenseNet-100 with k=24
    """
    return _densenet(DenseLayer, [32, 32, 32], 24, 1, pretrained, progress, **kwargs)

def densenetbc100_k12(pretrained=False, progress=False, **kwargs):
    """
    DenseNet-BC-100 with k=12
    """
    return _densenet(BottleneckLayer, [16, 16, 16], 12, 0.5, pretrained, progress, **kwargs)

def densenetbc250_k24(pretrained=False, progress=False, **kwargs):
    """
    DenseNet-BC-250 with k=24
    """
    return _densenet(BottleneckLayer, [41, 41, 41], 24, 0.5, pretrained, progress, **kwargs)

def densenetbc190_k40(pretrained=False, progress=False, **kwargs):
    """
    DenseNet-BC-190 with k=40
    """
    return _densenet(BottleneckLayer, [31, 31, 31], 40, 0.5, pretrained, progress, **kwargs)
