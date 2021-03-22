import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# instantiate deeplabv3_resnet101 with pretrained values
deeplab_model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=False, num_classes=3).to(
    device).eval()
# deeplab_model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=False).to(device).eval()

from torchvision import transforms
from PIL import Image
from torchvision import datasets
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from dataset_manager import RunwaysDataset
from torch.utils.data import Dataset, DataLoader
import wandb


class SemanticSegmentation(nn.Module):
    """
    Class to train and use a Semantic Segmentation Network
    """
    def __init__(self, model):
        super(SemanticSegmentation, self).__init__()
        self.model = model

    def forward(self, input_batch):
        """
        Runs the nn without backpropagation to see what it does when the values are unchanged

        :param input_batch:
        :return:
        """
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')
        with torch.no_grad():
            output = self.model(input_batch)['out'].to(device)
        return output
        # return output.argmax(0)  # returns the most likely label in a given region


def train_loop(dataloader, model, loss_fn, optimizer) -> None:
    size = len(dataloader)
    for i_batch, sample_batched in enumerate(dataloader):
        X = sample_batched['image']
        y = sample_batched['mask']
        # Compute prediction and loss
        pred = model(X)
        print(f"X.shape: {X.shape}, y.shape:{y.shape}, pred.shape:{pred.shape}")

        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_batch % 100 == 0:
            loss, current = loss.item(), i_batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            wandb.log({"loss": loss})


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            X = sample_batched['image']
            y = sample_batched['mask']
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}% Avg loss: {test_loss:>8f} \n")


def initialize_dataloader(dataset_name: str, labels: dict, batch_size: int = 4, num_workers: int = 1):
    """
    Generates a dataset object for to train or test the model

    :param dataset_name:
    :param labels:
    :param batch_size:
    :param num_workers:
    :return:
    """
    dirname = os.path.dirname(__file__)  # get the location of the root directory
    dirname = os.path.join(dirname, dataset_name)
    dataset = RunwaysDataset(dirname, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader


def model_pipeline(hyper_parameters, network_type):
    """
    Procedure to make train and test the CNN based upon the predefined hyperparameters
    :param hyper_parameters:
    :return:
    """
    with wandb.init(project="runway-segmentation", config=hyper_parameters):
        config = wandb.config  # log hyperparameters from config dict to wandb
        model, dataloader, criterion, optimizer = model_maker(config, network_type)  # make model with optimizer
        # and data
        # print(model)
        train_loop(dataloader, model, criterion, optimizer)  # train the model
        test_loop(dataloader, model, criterion)  # test the model
    return model


def model_maker(config, network_type):
    """
    Setup a CNN
    :param config: the hyper-parameters to encode the values with
    :return: nn.model, dataloader, loss_fn, optimizer with hyper-parameters
    """
    # Generate data
    dataloader = initialize_dataloader(config.dataset, category_rgb_vals, config.batch_size)

    # Setup model
    model = SemanticSegmentation(network_type)
    wandb.watch(model)

    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    return model, dataloader, criterion, optimizer


if __name__ == '__main__':

    # wandb.init(project="segmentation-tests")  # login to weights and biases

    default_config = dict(
        epochs=20,
        learning_rate=1e-3,
        batch_size=1,
        dataset="test_set",
        classes=3,
    )
    category_rgb_vals = {
        tuple([0, 0, 0]): 0,
        tuple([78, 53, 104]): 1,
        tuple([155, 47, 90]): 2
    }
    # deeplab_network = SemanticSegmentation(deeplab_model)
    #
    # model_pipeline(default_config, deeplab_network)

    category_rgb_vals = {
            tuple([0, 0, 0]): 0,
            tuple([78, 53, 104]): 1,
            tuple([155, 47, 90]): 2
    }

    category_rgb_names = {
        (0, 0, 0): "sky",
        (78, 53, 104): "runway",
        (155, 47, 90): "ground"
    }

    # Hyper-parameters
    learning_rate = 1e-3
    batch_size = 1
    epochs = 20

    # Initialize the network
    deeplab_network = SemanticSegmentation(deeplab_model)
    # wandb.watch(test_network)

    # Initialize the loss function
    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    sgd_optimizer = torch.optim.SGD(deeplab_network.parameters(), lr=learning_rate)

    # Initialize the datasets
    test_set = initialize_dataloader("test_set", category_rgb_vals, batch_size)

    # train NN
    train_loop(test_set, deeplab_network, cross_entropy_loss_fn, sgd_optimizer)


# # Generate colours for images
# palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
# colors = torch.as_tensor([i for i in range(3)])[:, None] * palette
# colors = (colors % 255).numpy().astype("uint8")
#
# # plot the semantic segmentation predictions of 21 classes in each color
# r = Image.fromarray(tensor_mask.byte().cpu().numpy()).resize(input_mask.size)
# r.putpalette(colors)
#
# plt.imshow(r)
# plt.show()

# segmentation_model = models.seg.deeplabv3_resnet101(pretrained=True).to(device).eval()
# print(f'Number of trainable weights in the segmentation model: {model}')
