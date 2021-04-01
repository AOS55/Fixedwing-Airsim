import os
import torch
import segmentation_models_pytorch as smp
from models import get_model
from config import NetworkConfig, category_rgb_vals
from dataset_manager import RunwaysDataset, split_dataset
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


class SemanticSegmentation(nn.Module):
    """
    Class to train and use a Semantic Segmentation Network
    """
    def __init__(self, model):
        super(SemanticSegmentation, self).__init__()
        self.model = model

    def forward(self, input_batch):
        """
        Runs the nn without backpropagation to get the output.

        :param input_batch: the X input of data to the NN
        :return: a tensor of probabilities of each label as defined by the classifier and tensor depth
        """
        if torch.cuda.is_available():
            input_batch = input_batch.to(device)
            self.model.to(device)
        # with torch.no_grad():
        output = self.model(input_batch)['out'].to(device)
        return output
        # return output.argmax(0)  # returns the most likely label in a given region


def train_loop(dataloader, model, loss_fn, optimizer, epoch) -> None:
    """
    Train the neural network with a given loss_fn and optimizer

    :param dataloader: a dataloader class that returns random batched image and mask tensor pairs
    :param model: the torch nn model to be trained
    :param loss_fn: the loss function used as to calculate the 'error' between X and y
    :param optimizer: the optimizer used to minimize the loss function
    :param epoch: number of times we have gone through the loop
    :return: None
    """
    size = len(dataloader)
    for i_batch, sample_batched in enumerate(dataloader):
        X = sample_batched['image']
        y = sample_batched['mask']
        # Compute prediction and loss
        pred = model(X)
        y = y.to(device)
        # print(f"X.shape: {X.shape}, y.shape:{y.shape}, pred.shape:{pred.shape}")
        loss = loss_fn(pred, y)
        writer.add_scalar("Loss/train", loss, i_batch)
        writer.add_histogram("Prediction Distribution/train", y, i_batch)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_batch % 100 == 0:
            loss, current = loss.item(), i_batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, epoch):
    """
    Train the neural network with a given loss_fn and optimizer

    :param dataloader: a dataloader class that returns random batched image and mask tensor pairs
    :param model: the torch nn model to be trained
    :param loss_fn: the loss function used as to calculate the 'error' between X and y
    :param epoch: number of times we have gone through the loop
    :return: None
    """
    size = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            X = sample_batched['image']
            y = sample_batched['mask']
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            writer.add_scalar("Loss/test", test_loss, i_batch)

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}% Avg loss: {test_loss:>8f} \n")
    writer.add_scalar("Accuracy/test", correct, epoch)
    writer.add_scalar("Avg Loss/test", test_loss, epoch)
    writer.add_images("Predicted Segmentation/test", y, epoch)

def initialize_dataloader(dataset_name: str, labels: dict, batch_size: int = 4, num_workers: int = 1):
    """
    Generates a dataset object for to train or test the model

    :param dataset_name: name of the dataset used with this run
    :param labels: the labels of each part of the image, the category_rgb_dict is what is used for this normally
    :param batch_size: the size of each batch used to train the agent
    :param num_workers: the number of workers (threads) used to do batching
    :return: train_set, test_set
    """
    dirname = os.path.dirname(__file__)  # get the location of the root directory
    dataset = dataset_name
    dirname = os.path.join(dirname, '../..')
    dirname = os.path.join(dirname, 'data/segmentation-datasets')
    dirname = os.path.join(dirname, dataset)
    dataset = RunwaysDataset(dirname, labels)
    split_data_set = split_dataset(dataset, 0.25)
    test_set = DataLoader(split_data_set['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers // 2)
    train_set = DataLoader(split_data_set['train'], batch_size=batch_size, shuffle=False, num_workers=num_workers // 2)
    return test_set, train_set


def model_pipeline(hyper_parameters: NetworkConfig, model: torch.nn, loss_fn: torch.optim, optimizer: torch.optim,
                   category_vals: dict) \
        -> None:
    """
    Procedure to make train and test the CNN based upon the predefined hyperparameters

    :param model: The nn model used within the controller
    :param loss_fn: The loss function used to train the nn
    :param optimizer: The optmizer used to train the nn
    :param category_vals: The rgb values used to breakup the images within the dataset
    :param hyper_parameters: dictionary with the key parameters to train the neural network
    :return:
    """
    # Initialize the datasets
    test_set, train_set = initialize_dataloader(hyper_parameters.dataset, category_vals,
                                                hyper_parameters.batch_size, )
    # run in a loop
    for e in range(hyper_parameters.epochs):
        print(f"Epoch {e + 1}\n-----------")
        train_loop(train_set, model, loss_fn, optimizer, e)
        test_loop(test_set, model, loss_fn, e)
        save_model(model, e, optimizer, hyper_parameters.run_name)
        writer.flush()
    print("Done!")


def save_model(model: torch.nn, epoch: int, optimizer: torch.optim, run_name: str) -> None:
    """
    Save the key model values at various points in time

    :param model: the torch nn model
    :param epoch: the number of epochs to run the model for
    :param optimizer: the optimizer used to train the nn model
    :param run_name: the name of the directory to store all the tensors from this run into
    """
    path = os.path.dirname(__file__)  # get the location of the root directory
    path = os.path.join(path, '../..')  # go to upper level
    path = os.path.join(path, 'runs/deeplab-runs')  # go to segmentation dataset
    path = os.path.join(path, run_name)  # go to specific model itself
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, 'models')
    if not os.path.isdir(path):
        os.mkdir(path)
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'paramaters': summary(model, model.shape)  # I don't know how to use this?
    }
    save_path = os.path.join(path, str(epoch) + '.pt')
    torch.save(state, save_path)


def save_tensorboards(run_name: str):
    """
    Setup the tensorboard writer and the tensorboard run directory

    :param run_name: the name of the directory to store the tensorboard within
    :return: the tensorboard summary writer object
    """
    path = os.path.dirname(__file__)  # get the location of the root directory
    path = os.path.join(path, '../..')  # go to upper level
    path = os.path.join(path, 'runs/deeplab-runs')  # go to segmentation dataset
    path = os.path.join(path, run_name)  # go to specific model itself
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, 'tensor_boards')
    if not os.path.isdir(path):
        os.mkdir(path)
    tb_writer = SummaryWriter(path)
    return tb_writer


if __name__ == '__main__':

    # Setup the nn configuration
    config = NetworkConfig()
    print(config.epochs)
    # Setup the device to use
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Found device: {device}")
    # Start the tensorboard summary writer
    writer = save_tensorboards(config.run_name)
    # Initialize the network
    model = config.model_name
    network = SemanticSegmentation(get_model(model, device))
    # Initialize the loss function
    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    sgd_optimizer = torch.optim.SGD(network.parameters(), config.learning_rate)
    # Train the model
    model_pipeline(config, network, cross_entropy_loss_fn, sgd_optimizer, category_rgb_vals)
    # Close the tensorboard summary writer
    writer.close()
