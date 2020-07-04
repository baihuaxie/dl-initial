""" test densenet.py """
import sys
import pytest
from torchsummary import summary

sys.path.append('..')

import model.densenet as net
import data_loader as dataloader
import utils


@pytest.fixture
def datadir():
    """ set directory containing dataset """
    return '../data/'


@pytest.fixture
def params():
    """ read params from json file """
    return utils.Params('../experiments/base-model/params.json')

@pytest.fixture
def select_data(datadir):
    """ select n random images + labels from train """
    return dataloader.select_n_random('train', datadir, n=2)

def test_densenet40_k12(select_data):
    """ test densenet40_k12 model """
    model = net.densenet40_k12()
    images, _ = select_data
    print(tuple(images.size()))
    output = model(images.float())
    utils.print_net_summary('./log_test', model, images)


