""" test mobilenet.py """

import sys
import pytest

sys.path.append('..')

import torch

import model.mobilenet as net
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

def test_mobilenet20_1p0_t3(select_data, device):
    """ test network layer=20, width_mult=1.0, expansion=3 """
    model = net.mobilenet20_1p0_t3().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_test_mobilenet20_1p0_t3', model, images)


def test_mobilenet20_1p0_t4(select_data, device):
    """ test network layer=20, width_mult=1.0, expansion=4 """
    model = net.mobilenet20_1p0_t4().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_test_mobilenet20_1p0_t4', model, images)

