""" test shufflenetv1.py """

import torch

import sys
import pytest

sys.path.append('..')

import model.shufflenetv2 as net
import data_loader as dataloader
import utils


@pytest.fixture
def datadir():
    """ set directory containing dataset """
    return '../data/'


@pytest.fixture
def device():
    """ if cuda is available """
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def params():
    """ read params from json file """
    return utils.Params('../experiments/base-model/params.json')


@pytest.fixture
def select_data(datadir, device):
    """ select n random images + labels from train """
    images, labels = dataloader.select_n_random('train', datadir, n=2)
    images, labels = images.to(device), labels.to(device)
    return images.float(), labels


def test_shufflenetv2_51_s0p5(select_data, device):
    """
    shufflenetv1
    - layers = 51
    - scaling = 0.5
    - channel_split = 0.5
    """
    model = net.shufflenetv2_51_s0p5().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_shufflenetv2_51_s0p5', model, images)


def test_shufflenetv2_51_s1p0(select_data, device):
    """
    shufflenetv1
    - layers = 51
    - scaling = 1.0
    - channel_split = 0.5
    """
    model = net.shufflenetv2_51_s1p0().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_shufflenetv2_51_s1p0', model, images)


def test_shufflenetv2_51_s1p5(select_data, device):
    """
    shufflenetv1
    - layers = 51
    - scaling = 1.5
    - channel_split = 0.5
    """
    model = net.shufflenetv2_51_s1p5().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_shufflenetv2_51_s1p5', model, images)


def test_shufflenetv2_51_s2p0(select_data, device):
    """
    shufflenetv1
    - layers = 51
    - scaling = 2.0
    - channel_split = 0.5
    """
    model = net.shufflenetv2_51_s2p0().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_shufflenetv2_51_s2p0', model, images)

