""" test net.py """

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pytest

import model.net as net
import model.data_loader as data_loader
import utils


@pytest.fixture
def datadir():
    """ set directory containing dataset """
    return './data/'

@pytest.fixture
def params():
    """ read params from json file """
    return utils.Params('./experiments/base-model/params.json')

@pytest.fixture
def fetch_train_batch(datadir, params):
    """ fetch a random batch from the trainset """
    dl = data_loader.fetch_dataloader(['train'], datadir, params)
    return iter(dl['train']).next()

@pytest.fixture
def Net(params):
    """ initialize a network object """
    return net.Net(params)

@pytest.fixture
def train_net_batch(Net, fetch_train_batch):
    """ train network on a random train batch """
    images = fetch_train_batch[0]
    outputs = Net(images)
    return outputs

@pytest.fixture
def compute_loss(fetch_train_batch, train_net_batch):
    """ compute the loss """
    outputs = train_net_batch
    labels = fetch_train_batch[1]
    return net.loss_fn(outputs, labels)


def test_net_is_nn_module_obj(Net):
    """ check Net() returns an nn.Module object """
    assert isinstance(Net, nn.Module), "- Net() type does not match"


def test_output_is_probabilities(train_net_batch, params):
    """ check model output is a probability distribution over labels """
    outputs = train_net_batch

    # check that outputs is a tensor obj
    assert isinstance(outputs, torch.Tensor), "- output type does not match"
    # check output sizes
    assert list(outputs.size()) == [params.batch_size, 
        10], "- output size does not match"
    # check output sums to batch_size*1
    assert int(np.round(torch.sum(torch.exp(outputs)).item())) == params.batch_size, "- outputs is not probability"


def test_xent_loss(compute_loss, train_net_batch, fetch_train_batch):
    """ check loss is cross-entropy """
    outputs = train_net_batch
    labels = fetch_train_batch[1]
    loss = compute_loss

    # check loss is a torch.Tensor obj
    assert isinstance(loss, torch.Tensor), "- loss type does not match"
    # check loss is the cross-entropy loss (reduction=mean; default)
    # use labels as indices to select from outputs, then sum / average
    # note that nn.CrossEntropyLoss() output is positive
    mask = torch.BoolTensor([[True if i == labels[j] else False 
        for i in range(outputs.size()[1])] for j in range(outputs.size()[0])])
    assert np.round(torch.sum(torch.masked_select(outputs, mask)).item() +
        loss.item() * outputs.size()[0], 2) == 0.00,  "- loss is not cross entropy"


def test_every_layer_updated_after_training(Net, fetch_train_batch):
    """ check model parameters are updated after one batch training """
    # configure Adam optimizer; use a large lr
    optimizer = optim.Adam(Net.parameters(), lr=10)
    # make a copy of network parameters before training
    before = [t.clone() for t in list(Net.parameters())]
    
    # do training on one batch
    images, labels = fetch_train_batch
    outputs = Net(images)
    optimizer.zero_grad()
    loss = net.loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    # make a copy of network parameter after training
    after = [t.clone() for t in list(Net.parameters())]

    # assert that any elements in the weights are updated by training
    for i in range(len(before)):
        assert (before[i] != after[i]).any(), "- layer {} not updated".format(i+1)




