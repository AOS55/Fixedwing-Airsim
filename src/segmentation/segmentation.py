import os
import torch
import segmentation_models_pytorch as smp
from models import get_model
from config import NetworkConfig
from dataset_manager import RunwaysDataset, split_dataset, CityscapesDataset
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
# from torchvision import datasets
# from torchsummary import summary
# import torch.autograd.profiler as profiler
from utils import save_model, initialize_tensorboards, tensor_to_image, tensor_image_to_image, startup_print, \
    get_sample_mean
import copy


class SemanticSegmentation(nn.Module):
    """
    Class to train and use a Semantic Segmentation Network
    """
    def __init__(self, model, device_type):
        super(SemanticSegmentation, self).__init__()
        self.model = model
        self.device = device_type

    def forward(self, input_batch):
        """
        Runs the nn without backpropagation to get the output.

        :param input_batch: the X input of data to the NN
        :return: a tensor of probabilities of each label as defined by the classifier and tensor depth
        """
        if torch.cuda.is_available():
            input_batch = input_batch.to(self.device)
            self.model.to(self.device)
        output = self.model(input_batch)
        if type(output) == torch.Tensor:
            return output.to(self.device)
        else:
            return output['out'].to(self.device)
        # return output.argmax(0)  # returns the most likely label in a given region


def initialize_dataloader(dataset_name: str, labels: dict, batch_size: int = 4, num_workers: int = 1,
                          class_name: str = 'runway', crop_size: tuple = (480, 852)):
    """
    Generates a dataset object for to train or validate the model

    :param dataset_name: name of the dataset used with this run
    :param labels: the labels of each part of the image, the category_rgb_dict is what is used for this normally
    :param batch_size: the size of each batch used to train the agent
    :param num_workers: the number of workers (threads) used to do batching
    :param class_name: the name of the class the segmented image is learning from, informs the type of Dataset class
    :param crop_size: an HxW scale of the desired crop e.g. 480x852
    :return: train_set, validation_set
    """
    dirname = os.path.dirname(__file__)  # get the location of the root directory
    dataset = dataset_name
    dirname = os.path.join(dirname, '../..')
    dirname = os.path.join(dirname, 'data/segmentation-datasets')
    dirname = os.path.join(dirname, dataset)
    if class_name == 'runway':
        dataset = RunwaysDataset(dirname, labels)
        split_data_set = split_dataset(dataset, 0.25)
        train_set = DataLoader(split_data_set['train'], batch_size=batch_size, shuffle=False,
                               num_workers=num_workers // 2, drop_last=True)
        validation_set = DataLoader(split_data_set['validation'], batch_size=batch_size, shuffle=False,
                              num_workers=num_workers // 2, drop_last=True)
    elif class_name == 'uav':
        train_dataset = RunwaysDataset(os.path.join(dirname, 'train'), labels, crop_size=crop_size)
        validation_dataset = RunwaysDataset(os.path.join(dirname, 'validation'), labels, crop_size=crop_size)
        train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers // 2)
        validation_set = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
                                                                                                        // 2)
    elif class_name == 'cityscapes':
        train_dataset = CityscapesDataset(dirname, labels, crop_size=crop_size, split_type='train')
        validation_dataset = CityscapesDataset(dirname, labels, crop_size=crop_size, split_type='val')
        train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers // 2)
        validation_set = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
                                                                                                        // 2)
    else:
        print(f'Invalid class_name: {class_name}')
        raise SystemExit(0)
    return train_set, validation_set


def model_pipeline(network_config: NetworkConfig, model: torch.nn,
                   loss_fn: torch.optim, optimizer: torch.optim, scheduler: lr_scheduler.StepLR) -> None:
    """
    Procedure to make, train and validate the CNN based upon the predefined hyperparameters

    :param network_config: dictionary with the key parameters to train the neural network
    :param model: The nn model used within the controller
    :param loss_fn: The loss function used to train the nn
    :param optimizer: The optmizer used to train the nn
    :param scheduler: The learning rate scheduler, to decay lr as epochs go by
    :return:
    """
    # Initialize the datasets
    train_set, validation_set = initialize_dataloader(network_config.dataset, network_config.classes,
                                                      network_config.batch_size, network_config.num_workers,
                                                      network_config.class_name,
                                                      (network_config.image_height, network_config.image_width))
    best_acc = 0
    best_e = 0
    best_model = copy.deepcopy(model)
    # optimize weights and biases
    for e in range(network_config.epochs):
        print(f"Epoch {e + 1}\n-----------")
        train_loop(train_set, model, loss_fn, optimizer, e)
        e_acc = validation_loop(validation_set, model, loss_fn, e, config.classes, config.batch_size)
        scheduler.step()  # Update the scheduler
        # Early stopping condition, only save the best model
        if e_acc > best_acc:
            best_acc = e_acc
            best_model = copy.deepcopy(model)
            best_e = e
        writer.flush()

    save_model(best_model, best_e, optimizer, network_config.run_name, network_config.dataset)
    print("Done!")


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
    train_loss, correct, jaccard_loss = 0, 0, 0
    for i_batch, sample_batched in enumerate(dataloader):
        X = sample_batched['image']
        y = sample_batched['mask']
        # Compute prediction and loss
        pred = model(X)
        y = y.to(device)
        # print(f"X.shape: {X.shape}, y.shape:{y.shape}, pred.shape:{pred.shape}")
        loss = loss_fn(pred, y)
        train_loss += loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item() / (y.shape[1] * y.shape[2])
        # jaccard_loss += smp.utils.functional.iou(pred, y).item()
        # writer.add_histogram("accuracy-distribution/train", (pred.argmax(1) == y).type(torch.float).sum().item()
        #                      / (y.shape[1] * y.shape[2]), i_batch)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss is {loss}")
        if i_batch % 100 == 0:
            loss, current = loss.item(), i_batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            # writer.add_graph(model, X)

    train_loss /= size
    correct /= size
    # jaccard_loss /= size
    writer.add_scalar("train/avg accuracy", correct, epoch)
    writer.add_scalar("train/avg loss", train_loss, epoch)
    # writer.add_scalar("train/avg IoU", jaccard_loss, epoch)


def validation_loop(dataloader, model, loss_fn, epoch, rgb_map: dict, batch_size: int) -> float:
    """
    Validate the neural network with a given loss_fn and optimizer

    :param dataloader: a dataloader class that returns random batched image and mask tensor pairs
    :param model: the torch nn model to be trained
    :param loss_fn: the loss function used as to calculate the 'error' between X and y
    :param epoch: number of times we have gone through the loop
    :param rgb_map: rgb_map used in original image
    :param batch_size: number of batches used to train on
    :return: avg_loss
    """
    size = len(dataloader)
    validation_loss, correct, jaccard_loss = [], [], []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            X = sample_batched['image']
            y = sample_batched['mask']
            pred = model(X)
            y = y.to(device)

            validation_loss.append(loss_fn(pred, y).item())
            correct.append((pred.argmax(1) == y).type(torch.float).sum().item() / (y.shape[1] * y.shape[2]))
            # jaccard_loss.append(smp.utils.functional.iou(pred, y).item())

            # writer.add_histogram("accuracy-distribution/train", (pred.argmax(1) == y).type(torch.float).sum().item()
            #                      / (y.shape[1] * y.shape[2]), i_batch)

            if i_batch % 20 == 1:
                pred_fig = tensor_to_image(pred, rgb_map, True)
                y_fig = tensor_to_image(y, rgb_map, False)
                X_fig = tensor_image_to_image(X)
                writer.add_figure('prediction/'+str(i_batch), pred_fig, global_step=epoch)
                writer.add_figure('truth/'+str(i_batch), y_fig, global_step=epoch)
                writer.add_figure('input/'+str(i_batch), X_fig, global_step=epoch)

    validation_mean = get_sample_mean(validation_loss)
    correct_mean = get_sample_mean(correct)
    # jarrard_mean = get_sample_mean(jaccard_loss)
    print(f"Validation Error: \n Accuracy: {(100 * correct_mean / batch_size):>0.1f}% Avg loss: "
          f"{validation_mean / batch_size:>8f} \n")
    writer.add_scalar("val/avg accuracy", correct_mean / batch_size, epoch)
    writer.add_scalar("val/avg loss", validation_mean / batch_size, epoch)
    # writer.add_scalar("val/avg IoU", jarrard_mean, epoch)
    # TODO: Jacquard-loss is doing something weird, IoU doesn't look correct
    return validation_mean


if __name__ == '__main__':
    # Setup the nn configuration
    config = NetworkConfig()
    startup_print(config)
    print(config.dataset)
    # Setup the device to use
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Found device: {device}")
    # Start the tensorboard summary writer
    writer, tb_path = initialize_tensorboards(config.run_name, config.dataset)
    # Initialize the network
    # with profiler.profile() as prof:  # profile network_initialization
    #     with profiler.record_function("network_initialization"):
    network_model = config.model_name
    pretrained = config.pretrained
    model = SemanticSegmentation(get_model(network_model, device, (len(config.classes) + 1), pretrained), device)
    if torch.cuda.device_count() > 1:
        print(f"Lets use {torch.cuda.device_count()}, GPUs!")
        model = nn.DataParallel(model).to(device)
    model.to(device)

    # prof.export_chrome_trace(os.path.join(tb_path, "network_trace.json"))
    # Initialize the loss function
    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    # setup the optimizer
    sgd_optimizer = torch.optim.SGD(model.parameters(), config.learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(sgd_optimizer,
                                           step_size=config.lr_scheduler_step_size,
                                           gamma=config.lr_depreciation)
    # Train the model
    # TODO: Profiler is here but it is too big, try and profile individual processes once
    # with profiler.profile() as prof:  # profile training and validation process
    #     with profiler.record_function("learning"):
    model_pipeline(config, model, cross_entropy_loss_fn, sgd_optimizer, exp_lr_scheduler)
    # prof.export_chrome_trace(os.path.join(tb_path, "learning_trace.json"))
    # Close the tensorboard summary writer
    writer.close()
