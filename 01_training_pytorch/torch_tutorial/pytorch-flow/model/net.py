"""
    Defines the neural network, loss function and metrics

    Network:
    - define a custom class net()
    - inherits nn.Module
    - takes hyperparameters as arguments to the class
    - implement a __init__() method -> contains layers / modules used by the network
        - usually put layers with learnable weights in __init__() method
        - parameterize the arguments to each layer instance as much as possible
    - implement a forward(x) method -> starts from input tensor x and applies each layer till an output tensor is produced
        - use nn.functional methods to apply activations, pooling, etc.

    loss function:
    - takes as input the model outputs and labels
        - by choice accept both train outputs and labels as tensors
    - returns a scalar as the loss
        - must return a tensor or other class objects with a backward() method to propagate gradients

    metrics:
    - a dictionary containing metrics to be used to monitor training progress
    - for each metric need to define a corresponding function to compute the value for the metrics
        - takes as arguments the model outputs, labels, etc.
    - by choice all metrics functions accepts tensors as inputs and returns floats or np.ndarray of floats

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 


class Net(nn.Module):
    """
    Standard way to define neural network in PyTorch:

    - define a custom model class which inherits the nn.Module class

    - choose the layers(modules) in the __init__() method
        - __init__ first inherits (super) the contructor from nn.Module class
        - each layer is passed to a self.layer field
        - note that layer input/output dimensions must match between consecutive layers
        - usually only define layers with learnable weights (linear, conv, fc, bn, etc.) in __init__() method

    - define a forward() method that applies the layers as a stack on the input to produce an output
        - takes as input a tensor x -> the input to be passed throught the neural network
            - note that input tensor dimension must match first layer in the network
        - usually follows a syntax such as x = self.layer(x) to apply each layer
            - for complex layers that have diverging paths, may add merging layers at its output and use syntax such as x1 = self.layer(x)
        - use nn.functional class to apply parameter-free modules, e.g., pooling, activations, etc., in between layers
        - returns the raw output as designed by the network

    """

    def __init__(self, params):
        """
        Defines the layers used in the network

        - 2 convolutional layers
        - 3 fc layers
        - 4 bn layers
        
        Args:
            params: (Params) contains num_channels, dropout_rate

        """

        super(Net, self).__init__()
        self.num_channels = params.num_channels

        # each convolutional layer has arguments of (input_channels, output_channels, kernel_size, stride=1, padding=0)
        # each batch_normalization layer has arguments of (input_channels)
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # each fc layer has arguments of (input_activations, output_activations)
        # first fc layer should take input from converted (flatten) layer that flattens the convolutional output tensor
        # this is also the only fc layer that has input arguments depending on implementation in the forward() method
        self.fc1 = nn.Linear(4*4*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 10)
        self.dropout_rate = params.dropout_rate

    def forward(self, x: torch.tesor) -> torch.tensor:
        """
        Defines how to use the layers to operate on an input tensor.

        Args:
            x: (tensor) contains a batch of images, of dimension = [batch_size, 3, 3, 32]

        Returns:
            out: (tensor) dimension of [batch_size, 10] with log probabilities for the labels of each image

        """
        #                                                       # batch_size x 3 x 32 x 32
        # apply conv + batch_norm + max_pool + relu
        # each max_pool layer has arguments of (input, stride)
        x = self.bn1(self.conv1(x))                             # batch_size x num_channels x 32 x 32  -> first increase channel dimension
        x = F.relu(F.max_pool2d(x), 2)                          # batch_size x num_channels x 16 x 16  -> second decrease activation dimension
        x = self.bn2(self.conv2(x))                             # batch_sze x num_channels*2 x 16 x 16
        x = F.relu(F.max_pool2d(x, 2))                          # batch_size x num_channels*2 x 8 x 8
        x = self.bn3(self.conv3(x))                             # batch_size x num_channels*4 x 8 x 8 
        x = F.relu(F.max_pool2d(x, 2))                          # batch_size x num_channels*4 x 4 x 4

        # flatten the activations
        x = x.view(-1, self.num_flat_features(x))               # batch_size x 4*4*num_channels*4

        # apply fc + fc_batch_norm + relu + dropout
        # last fc layer outputs un-normalized logits
        x = F.dropout(F.relu(self.fcbn1(self.fc1(x))), 
                p=self.dropout_rate, training=self.training)    # batch_size x num_channels*4
        x = self.fc2(x)                                         # batch_siz x 10 -> # of class labels on cifar-10

        # apply log softmax to return normalized probabilites
        return F.log_softmax(x, dim=1)

    @staticmethod
    def num_flat_features(x: torch.tensor) -> int:
        """
        computes dimension for falltened activations

        Args:
            x: (tensor) tensor to be flattened

        Returns:
            num_features: (int) interger = # of flattened activations = channels*activations

        Note: this method doesn't reference the class Net() itself, so should be declared as a static method

        """

        # first axis = batch_size, not flattened
        size = x.size()[1:]
        num_features = 1
        for s_dim in size:
            num_features *= s_dim
        return num_features


def loss_fn(outputs: torch.tensor, labels: np.ndarray) -> nn.CrossEntropyLoss:
    """
    Computes the cross entropy loss given the predicted log probabilites and labels

    Args:
        outputs: (tensor) dimension = batch_size x 10 -> each element in batch_size dim is a list containing
                  the predicted log probabilities for 10 classes for the image
        labels: (ndarray) dimen = batch_size x 1 -> each element is an integer representing the correct label

    Returns:
        loss: (nn.CrossEntroyLoss) dimension = 1 (scalar) -> a tensor containing the cross entropy loss over the batch

    Note:
        - returned loss must be a class object with a backward() method to facilitate backpropagation
        - can be a torch.tensor with requires_grad=True

    """

    return nn.CrossEntropyLoss(outputs, torch.from_numpy(labels))


def accuracy(outputs: torch.tensor, labels: torch.tensor) -> float:
    """
    Computes the classification accuracy metric

    Args:
        outputs: (tensor) dimension = batch_size x 10 -> each element in batch_size dim is a list containing the predicted log probabilities for 10 classes for the image 
        labels: (tensor) dimen = batch_size x 1 -> each element is an integer representing the correct label

    Returns:
        accuracy: (float) accuracy in [0,1]

    """
    
    _, predicted_labels = torch.max(outputs, dim=1)
    return np.sum(predicted_labels == labels.numpy()) / float(labels.numpy().size)


# maitain all metrics used in the training and evaluation loops in this dictionary
metrics = {
    'accuracy' : accuracy,
}