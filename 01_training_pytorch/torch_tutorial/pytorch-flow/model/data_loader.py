"""
    Reads raw dataset and returns dataloader(s)

    This file should take the following as inputs:
    - raw dataset -> url for download; file path for local
    - hyper-parameters, e.g., num_workers, pinning memeory, batch_size, etc.

    This file should return:
    - torch.utils.data.DataLoader objects -> conceptually, should distinct dataloaders for train, val, test datasets

    The following items need to be implemented in this file:

    1) a class definition for dataset
    -- if using pytorch built-in datasets, e.g., torchvision.datasets, can directly call the dataset class and omit custom definition
    -- if using custom datasets, need to define a class for that dataset
       - class needs to inherit from either Dataset or IterableDataset pytorch classes
       - for Dataset class, need to define __getitem__() and __len__() methods
       - for IterableDataset class, need to define __iter__() method
    -- class input: directory, tansforms
       - needs to apply transforms to raw dataset before return

    2) a set of transform pipelines
    -- for each dataloader type (train, val, test, etc.), may need to define different transform pipelines
    -- usually uses torchvision.transforms.Compose() to construct the pipelines

    3) a function / method to return the corresponding DataLoader object for each type
    -- need to instantiate the defined or built-in dataset class
    -- usually uses torch.utils.data.DataLoader() to build DataLoader objects
    -- this function / method is then called in main scripts to load dataset

"""

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# transforms pipeline for train set
train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# transforms pipeline for test set
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)


def fetch_dataloader(types, datadir, params):
    """
    Fetches the dataloader objects for each type from datadir.

    Args:
        types: (list) has one or more of "train, val, test" depending on the dataset
        datadir: (str) file path containing the raw dataset
        params: (Params object) hyperparameters controlling data loading behavior

    Returns:
        dataloadr: (dict) contains the DataLoader objects for each type in types
    """

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:

            # apply train set tranforms if train data
            if split == 'train':
                trainset = CIFAR10(datadir, download=False, train=True, transform=train_transform)
                dataloader = DataLoader(trainset, batch_size=params.batch_size, shuffle=True, 
                            num_workers=params.num_works,
                            pin_memory=params.pin_cuda)

            # apply test set transforms if test data
            if split == 'test':
                testset = CIFAR10(datadir, download=False, train=False, transform=test_transform)
                dataloader = DataLoader(testset, batch_size=params.batch_size, shuffle=True, 
                            num_workers=params.num_workers,
                            pin_memory=params.pin_cuda)
            
            dataloaders[split] = dataloader

        return dataloaders
                        
