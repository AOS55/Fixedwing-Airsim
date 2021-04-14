import torch
import os
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import math


def save_model(model_state_dict: dict, epoch: int,
               optimizer_state_dict: dict,
               run_name: str,
               dataset_name: str) -> None:
    """
    Save the key model values at various points in time

    :param model_state_dict: the weights and biases within the nn
    :param epoch: the number of epochs to run the model for
    :param optimizer_state_dict: the optimizer hyper-parameters
    :param run_name: the name of the directory to store all the tensors from this run into
    :param dataset_name: the name of the dataset for this experiment
    :return: None
    """
    path = os.path.dirname(__file__)  # get the location of the root directory
    path = os.path.join(path, '../..')  # go to upper level
    path = os.path.join(path, 'runs/deeplab-runs')  # go to segmentation dataset
    path = os.path.join(path, dataset_name)  # go to runs dataset dir
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, 'models')
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, run_name)
    if not os.path.isdir(path):
        os.mkdir(path)
    state = {
        'epoch': epoch,
        'state_dict': model_state_dict,
        'optimizer': optimizer_state_dict,
        # 'paramaters': summary(model, model.shape)  #TODO: I don't know how to use summary module?
    }
    save_path = os.path.join(path, str(epoch) + '.pt')
    torch.save(state, save_path)


def initialize_tensorboards(run_name: str, dataset_name: str):
    """
    Setup the tensorboard writer and the tensorboard run directory

    :param run_name: the name of the directory to store the tensorboard
    :param dataset_name: the name of the dataset for this experiment
    :return: the tensorboard summary writer object & path to the tb file
    """
    path = os.path.dirname(__file__)  # get the location of the root directory
    path = os.path.join(path, '../..')  # go to upper level
    path = os.path.join(path, 'runs/deeplab-runs')  # go to segmentation dataset
    path = os.path.join(path, dataset_name)  # go to runs dataset dir
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, 'tensor_boards')
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, run_name)  # go to specific model dir
    ver = 0
    path_tmp = path + 'ver' + str(ver)
    while os.path.exists(path_tmp):
        ver += 1
        path_tmp = path + 'ver' + str(ver)
    path = path_tmp
    if not os.path.isdir(path):
        os.mkdir(path)
    tb_writer = SummaryWriter(path)
    return tb_writer, path


def tensor_to_image(input_tensor: torch.tensor,  rgb_map: dict, is_prediction: bool = False) -> \
        plt.figure:
    """
    Given an input image produces a coloured segmented response for visualization

    :param input_tensor: an input image with dims as expected by the nn model
    :param rgb_map: rgb_map used in original image
    :param is_prediction: a boolean of whether the data is a CXWXH tensor or just 1XWXH
    :return: matplotlib figure object
    """
    # TODO: the difference in RGB input is an issue here as need to use rgb_map[color] for cityscapes
    if is_prediction:
        image_tensor = input_tensor.argmax(1)[0]
    else:
        image_tensor = input_tensor[0]
    # create a color palette, selecting a color for each class
    colors = []
    # TODO: This approach is likely slow, not sure how to improve without passing in dict type
    if [type(k) for k in rgb_map.keys()][0] == int:  # if dict is of type int: tuple(list(rgb_triplet)
        for color in rgb_map:
            colors.append(list(rgb_map[color]))
        colors = np.asarray(colors).astype('uint8')
    else:  # if dict is of type tuple(list(rgb_triplet): int
        for color in rgb_map:
            colors.append(list(color))
        colors = np.asarray(colors).astype('uint8')
    # plot the semantic segmentation predictions for each color
    r = Image.fromarray(image_tensor.byte().cpu().numpy())
    r.putpalette(colors)

    fig, ax = plt.subplots()
    ax.imshow(r)
    return fig


def tensor_image_to_image(input_tensor: torch.tensor) -> plt.figure:
    """
    Given an input image tensor converts the image to a matplotlib object to view

    :param input_tensor: input of the original image (X) tensor
    :return: a matplotlib image
    """

    # convert image from GPU to CPU and use first image
    img = input_tensor[0].cpu().numpy()
    # rearrange order to HWC
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    fig, ax = plt.subplots()
    ax.imshow(img)
    return fig


def startup_print(config):
    print(textwrap.dedent(f"""\
                Starting segmentation training with following parameters:
                DATASET_NAME: {config.dataset}
                CLASS_NAME: {config.class_name}
                RUN_NAME: {config.run_name}
                EPOCHS: {config.epochs}
                BATCH SIZE: {config.batch_size}
                LEARNING RATE: {config.learning_rate}
                LR DEPRECIATION [GAMMA]: {config.lr_depreciation}
                LR STEPS: {config.lr_scheduler_step_size}
                IMAGE HEIGHT: {config.image_height}
                IMAGE WIDTH: {config.image_width}
                DEVICE: {config.device}
                NUMBER OF WORKERS: {config.num_workers}
                """))


def get_sample_mean(values: list) -> float:
    """
    Calculates the sample mean (overline x) of the elements in a list

    :param values: list of values
    :return: sample mean
    """
    sample_mean = sum(values) / len(values)
    return sample_mean


def get_sample_variance(values: list) -> float:
    """
    Calculates the sample variance (S^2) of the elements in a list

    :param values: list of values
    :return: sample variance
    """
    sample_mean = get_sample_mean(values)
    sample_variance = sum([x - sample_mean for x in values]) / (len(values) - 1)
    return sample_variance


def get_student_t(values: list, representative_mean: float) -> float:
    """
    Calculate one-sample t-test for student's t distribution of CI-bounds, this assumes a gaussian with more samples.
    By central limit therom this test assumes sample mean is normal (don't know if that is true)

    :param values: list of values
    :param representative_mean: the mean expected from a large distribution, that is gaussian
    :return: one-sample student t-test
    """
    t = (get_sample_mean(values) - representative_mean) / math.sqrt(get_sample_variance(values) / len(list))
    return t


def get_gaussian_interval(values: list, z_score: float = 1.96) -> float:
    """
    Calculate the confidence interval if the mean of the values is representative of a gaussian distribution

    :param values: list of values
    :param z_score: the corresponding value of Phi(z) from a standard-normal distribution (mu=0 2sigma=1) 95% by default
    :return: sample variance
    """
    sample_mean = get_sample_mean(values)
    interval = z_score * math.sqrt(abs((sample_mean * (1 - sample_mean)) / len(values)))
    return interval


# Think very carefully about this section


# def get_wilson_score(values: list) -> float:
#     """
#     Calculate the wilson score, a CI bound for the probability of successes calculated from the outcome of a series
#     of Bernoulli trials.
#
#     ...
#     Binomial proportion confidence intervals assume samples are drawn from a binomial distribution. Where the
#     probability of binomial success
#     """

