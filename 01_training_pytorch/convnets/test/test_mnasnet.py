""" test mnasnet.py """

import sys
import pytest

sys.path.append('..')

import torch

import model.mnasnet as net
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

def test_mnasneta1(select_data, device):
    """ test mnasneta1 """
    model = net.mnasneta1().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_test_mnasneta1', model, images)


