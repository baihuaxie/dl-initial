""" test data_loader.py """

import torch
import numpy as np 

import pytest
import utils
import model.data_loader as data_loader

@pytest.fixture
def datadir():
    """ set directory containing dataset """
    return './data/'

@pytest.fixture
def params():
    """ read params from json file """
    return utils.Params('./experiments/base-model/params.json')

@pytest.fixture
def fetch_data(datadir, params):
    """ fetch the dataset dataloaders """
    dl = data_loader.fetch_dataloader(['train', 'test'], datadir, params)
    return dl

def test_return_dict(fetch_data):
    """ check if fetch_dataloader returns a dictionary """
    dl = fetch_data
    assert isinstance(dl, dict), "- return is not a dict"
    assert 'train' in dl.keys(), "- returned dict does not contain train"
    assert 'test' in dl.keys(), "- returned dict does not contain test"

def test_return_train_dataloader_obj(fetch_data):
    """ check if fetch_dataloader returns a DataLoader obj for trainset """
    dl = fetch_data
    assert isinstance(dl['train'], 
        torch.utils.data.DataLoader), "- return is not valid train dataloader"

def test_return_test_dataloader_obj(fetch_data): 
    """ check if fetch_dataloader returns a DataLoader obj for testset """   
    dl = fetch_data
    assert isinstance(dl['test'],
        torch.utils.data.DataLoader), "- return is not valid test dataloader"

def test_train_dl_length(fetch_data, params):
    """ check train dl contains full cifar10 trainset """
    dl = fetch_data
    # len(dataloader) * batch_size = # of training images
    # cifar-10 training set contains 50,000 3x32x32 images
    assert len(dl['train']) == int(np.ceil(50000/params.batch_size)), "- trainset size does not match"

def test_random_train_data_is_valid(fetch_data, params):
    """ check a random data in train dl matches cifar10 train image """
    dl = fetch_data
    train_data, train_labels = dl['train'].__iter__().next()
    
    # data: (torch.tensor) dimension = batch_size x 3 x 32 x 32
    assert list(train_data.size()) == [params.batch_size, 3, 32, 32]
    # labels: (torch.tensor) dimension = batch_size x 1
    assert list(train_labels.size()) == [params.batch_size]

def test_test_dl_length(fetch_data, params):
    """ check test dl contains full cifar10 testset """
    dl = fetch_data
    # cifar-10 test set contains 10,000 3x32x32 images
    assert len(dl['test']) == int(np.ceil(10000/params.batch_size)), "- testset size does not match"

def test_random_test_data_is_valid(fetch_data, params):
    """ check a random data in test dl matches cifar10 test image """
    dl = fetch_data
    test_data, test_labels = dl['test'].__iter__().next()
    assert list(test_data.size()) == [params.batch_size, 3, 32, 32]
    assert list(test_labels.size()) == [params.batch_size]


@pytest.fixture
def fetch_subset_data(datadir, params):
    """ fetch the subset dataloaders """
    num = 10
    dl = data_loader.fetch_subset_dataloader(
        ['train', 'test'], datadir, params, num)
    return dl, num

def test_train_subset_dataloader_len(fetch_subset_data, params):
    """ check the size of train subset dataloader """
    dl, num = fetch_subset_data
    assert len(dl['train']) == num, "- train subset dataloader size wrong"
