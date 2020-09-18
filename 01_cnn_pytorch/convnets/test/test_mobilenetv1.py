""" test mobilenetv1.py """

import sys
import pytest

# add parent directory to search path for 'import'
sys.path.append('..')

import torch

import model.mobilenetv1 as net
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


def test_mobilenetv1_28_1p0_32(select_data, device):
    """ test network structure of mobilenetv1 28 layers 1.0/32 """
    model = net.mobilenetv1_28_1p0_32().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_test_mobilenetv1_28_1p0_32', model, images)


def test_mobilenetv1_28_0p75_32(select_data, device):
    """ test network structure of mobilenetv1 28 layers 0.75/32 """
    model = net.mobilenetv1_28_0p75_32().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_test_mobilenetv1_28_0p75_32', model, images)


def test_mobilenetv1_28_1p25_32(select_data, device):
    """ test network structure of mobilenetv1 28 layers 1.25/32 """
    model = net.mobilenetv1_28_1p25_32().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_test_mobilenetv1_28_1p25_32', model, images)
