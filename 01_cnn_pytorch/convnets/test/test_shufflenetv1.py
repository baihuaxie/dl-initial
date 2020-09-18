""" test shufflenetv1.py """

import torch

import sys
import pytest

sys.path.append('..')

import model.shufflenetv1 as net
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


def test_shufflenetv1_50_s1p0_g1(select_data, device):
    """
    shufflenetv1
    - layers = 50
    - scaling = 1.0
    - groups = 1
    """
    model = net.shufflenetv1_50_s1p0_g1().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_shufflenetv1_50_s1p0_g1', model, images)


def test_shufflenetv1_50_s1p0_g2(select_data, device):
    """
    shufflenetv1
    - layers = 50
    - scaling = 1.0
    - groups = 2
    """
    model = net.shufflenetv1_50_s1p0_g2().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_shufflenetv1_50_s1p0_g2', model, images)


def test_shufflenetv1_50_s1p0_g3(select_data, device):
    """
    shufflenetv1
    - layers = 50
    - scaling = 1.0
    - groups = 3
    """
    model = net.shufflenetv1_50_s1p0_g3().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_shufflenetv1_50_s1p0_g3', model, images)


def test_shufflenetv1_50_s1p0_g4(select_data, device):
    """
    shufflenetv1
    - layers = 50
    - scaling = 1.0
    - groups = 4
    """
    model = net.shufflenetv1_50_s1p0_g4().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_shufflenetv1_50_s1p0_g4', model, images)


def test_shufflenetv1_50_s1p0_g8(select_data, device):
    """
    shufflenetv1
    - layers = 50
    - scaling = 1.0
    - groups = 8
    """
    model = net.shufflenetv1_50_s1p0_g8().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_shufflenetv1_50_s1p0_g8', model, images)


def test_shufflenetv1_50_s0p5_g1(select_data, device):
    """
    shufflenetv1
    - layers = 50
    - scaling = 0.5
    - groups = 1
    """
    model = net.shufflenetv1_50_s0p5_g1().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_shufflenetv1_50_s0p5_g1', model, images)


def test_shufflenetv1_50_s0p5_g2(select_data, device):
    """
    shufflenetv1
    - layers = 50
    - scaling = 0.5
    - groups = 2
    """
    model = net.shufflenetv1_50_s0p5_g2().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_shufflenetv1_50_s0p5_g2', model, images)


def test_shufflenetv1_50_s0p5_g3(select_data, device):
    """
    shufflenetv1
    - layers = 50
    - scaling = 0.5
    - groups = 3
    """
    model = net.shufflenetv1_50_s0p5_g3().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_shufflenetv1_50_s0p5_g3', model, images)


def test_shufflenetv1_50_s0p5_g4(select_data, device):
    """
    shufflenetv1
    - layers = 50
    - scaling = 0.5
    - groups = 4
    """
    model = net.shufflenetv1_50_s0p5_g4().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_shufflenetv1_50_s0p5_g4', model, images)


def test_shufflenetv1_50_s0p5_g8(select_data, device):
    """
    shufflenetv1
    - layers = 50
    - scaling = 0.5
    - groups = 8
    """
    model = net.shufflenetv1_50_s0p5_g8().to(device)
    images, _ = select_data
    _ = model(images)
    utils.print_net_summary('./log_shufflenetv1_50_s0p5_g8', model, images)

