""" Train the model """

# python
import argparse
import os
import logging

# torch
import torch
import torch.optim as optim

# custom
import model.net as net
import model.data_loader as data_loader
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='C:/Users/bhxie/Documents/Github/dl-initial/01_training_pytorch/data/cifar-10-batches-py',
                    help='Directory containing the dataset')
parser.add_argument('--model_dir', default='./experiments/base_model',
                    help='Directory containing the params.json')
parser.add_argument('--restore_file', default=None,
                    help='Optional, name of file in --model_dir containing weights / hyperparameters to be loaded before training')


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """
    Train the model on num_steps batches

    """
    pass


def train_and_evaluate(model, optimizer, train_loader, val_loader, loss_fn, metrics, params, 
                       model_dir, restore_file=None):
    """
    Train the model and evaluate on every epoch

    Args:
        model: (inherits torch.nn.Module) the custom neural network model
        optimizer: (inherits torch.optim) optimizer to update the model parameters
        train_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches the training set 
        val_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches the validation set
        loss_fn : (function) a function that takes batch_output (tensor) and batch_labels (np.ndarray) and return the loss (tensor) over the batch
        metrics: (dict) a dictionary of functions that compute a metric using the batch_output and batch_labels 
        params: (Params) hyperparameters
        model_dir: (string) directory containing params.json, learned weights, and logs
        restore_file: (string) optional = name of file to restore training from -> no filename extension .pth or .pth.tar/gz

    """
    pass


if __name__ == '__main__':

    # Load the params from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set random seed for reproducible experiments
    torch.manual_seed(200)
    if params.cuda:
        torch.cuda.manual_seed(200)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info('Loading datasets...')

    # Fetch the data loaders
    data_loaders = data_loader.fetch_dataloader(
        ['train', 'test'], args.data_dir, params
    )
    train_loader = data_loaders['train']
    test_loader = data_loaders['test']

    logging.info('- done.')

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    # Define the optimizer
    optimizer = optim.Adam(model.paramters(), lr=params.learning_rate)

    # Fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info('Starting training for {} epoch(s)...'.format(params.num_epochs))
    train_and_evaluate(model, optimizer, train_loader, test_loader, loss_fn, metrics, params,
                       args.model_dir, args.restore_file)