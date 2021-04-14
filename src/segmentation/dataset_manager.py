import os
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
# import matplotlib.pyplot as plt


class RunwaysDataset(Dataset):
    """
    Prepare input and training data from the UE4 segmentation maps
    """

    def __init__(self,
                 root_dir: str,
                 labels: dict,
                 semantic_dir: str = "segmentation_masks",
                 image_dir: str = "images",
                 crop_size: tuple = (480, 832)):
        """
        Args:
        :param root_dir: Root directory of all dataset files
        :param semantic_dir: path to directory containing all semantic images
        :param image_dir: path containing all raw images
        :param labels: dictionary containing the rgb triplets and labels of the image mask
        """
        self.root_dir = root_dir
        self.labels = labels
        self.semantic_dir = semantic_dir
        self.image_dir = image_dir
        self.crop_size = crop_size

    def __len__(self):
        """
        Get the number of frames in the dataset
        :return: simply the number of files in the semantic image directory
        """
        dirname = os.path.join(self.root_dir, self.semantic_dir)
        return len([name for name in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, name))])

    def __getitem__(self, idx):
        """
        Get samples from the dataset
        :param idx:
        :return: dict of tensors image and mask
        """
        mask_loc = os.path.join(self.root_dir, self.semantic_dir, str(idx) + ".png")
        image_loc = os.path.join(self.root_dir, self.image_dir, str(idx) + ".png")
        input_mask = Image.open(mask_loc)
        input_image = Image.open(image_loc)
        input_image, input_mask = apply_random_crop(input_image, input_mask, self.crop_size)
        input_image, input_mask = apply_random_flip(input_image, input_mask)
        mask_tensor = self.mask_preparation(input_mask)
        image_tensor = self.image_preparation(input_image)
        sample = {'image': image_tensor, 'mask': mask_tensor}
        return sample

    def mask_preparation(self, input_mask: Image.Image):
        """
        Takes an example of the pretrained mask and turns it into a tensor based on labels available.
        Based on the labels dictionary, remaps values of (rgb) triplets from the training set to use integer values.

        :param input_mask:
        :return: torch_mask: torch.Tensor of int values from dict, input_mask: PIL image type for resizing and outting
        """
        np_mask = np.array(input_mask)

        np_mask_labels = np.ones([len(np_mask), len(np_mask[0])])
        for col_id in range(np_mask.shape[0]):
            for row_id in range(np_mask.shape[1]):
                try:
                    np_mask_labels[col_id, row_id] = self.labels[tuple(np_mask[col_id, row_id])]
                except KeyError:
                    # print(f"Unrecognized value in mask data {np_mask[col_id, row_id]} setting to 0")
                    np_mask_labels[col_id, row_id] = 0
        torch_mask = torch.tensor(np_mask_labels)
        torch_mask = torch_mask.to(dtype=int)
        return torch_mask

    @staticmethod
    def image_preparation(input_image: Image.Image):
        """
        Takes a regular png image and prepares it to be an input to the CNN

        :param input_image: the PIL type input_image
        :return: mini_batch the tensor representation of images expected by the model
        """
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(input_image)
        return input_tensor


class CityscapesDataset(RunwaysDataset):
    """
    Loads in the cityscapes dataset using the dataloader technique
    """
    def __init__(self, root_dir: str, labels: dict, crop_size: tuple = (480, 852), split_type: str = 'validation'):
        super().__init__(root_dir, labels)
        self.root_dir = root_dir
        self.labels = labels
        self.crop_size = crop_size
        self.cityscape_set = datasets.Cityscapes(root_dir, split=split_type, mode='fine',
                                                 target_type='semantic')

    def __len__(self):
        return len(self.cityscape_set)

    def __getitem__(self, idx):
        input_image = self.cityscape_set[idx][0]
        input_mask = self.cityscape_set[idx][1]
        input_image, input_mask = apply_random_crop(input_image, input_mask, self.crop_size)
        mask_tensor = torch.tensor(np.array(input_mask)).to(dtype=int)
        image_tensor = self.image_preparation(input_image)
        sample = {'image': image_tensor, 'mask': mask_tensor}
        return sample


def apply_random_crop(img: Image.Image, tgt: Image.Image, crop_size: tuple = (480, 852)):
    """
    Apply a randomly located crop equally to the target and the image

    :param img: a PIL Image of the RGB image
    :param tgt: a PIL Image of the target image
    :param crop_size: an HxW scale of the desired crop e.g. 480x852
    :return: the cropped img & tgt PIL images
    """
    t = transforms.RandomResizedCrop(crop_size)
    state = torch.get_rng_state()
    img = t(img)
    torch.set_rng_state(state)
    tgt = t(tgt)
    return img, tgt


def apply_random_flip(img: Image.Image, tgt: Image.Image, prob: float = 0.5):
    """
    Apply a horizontal flop to target and image with a given probability

    :param img: a PIL Image of the RGB image
    :param tgt: a PIL Image of the target image
    :param prob: the probability of flipping and image
    :return: possibly flipped image & tgt PIL images
    """
    t = transforms.RandomHorizontalFlip(prob)
    state = torch.get_rng_state()
    img = t(img)
    torch.set_rng_state(state)
    tgt = t(tgt)
    return img, tgt


def split_dataset(dataset, split_size: float = 0.25):
    """
    Train valid split

    :param dataset: the dataset to be split
    :param split_size: the validation to train split
    """
    train_idx_list, validation_idx_list = train_test_split(list(range(len(dataset) - 1)), test_size=split_size)
    split_set = {'train': Subset(dataset, train_idx_list), 'validation': Subset(dataset, validation_idx_list)}
    return split_set


if __name__ == '__main__':

    # dir_name = os.path.dirname(__file__)  # get the location of the root directory
    # dataset = "480-multicct"
    # dir_name = os.path.join(dir_name, '../..')
    # dir_name = os.path.join(dir_name, 'data/segmentation-datasets')
    # dir_name = os.path.join(dir_name, dataset)
    # runway_rgb_vals = {
    #     tuple([0, 0, 0]): 0,
    #     tuple([78, 53, 104]): 1,
    #     tuple([155, 47, 90]): 2
    # }
    # data_set = RunwaysDataset(dir_name, runway_rgb_vals)
    # split_data_set = split_dataset(data_set, 0.25)
    #
    # train_dataloader = DataLoader(split_data_set['train'], batch_size=4, shuffle=False, num_workers=20)
    # validation_dataloader = DataLoader(split_data_set['validation'], batch_size=4, shuffle=False, num_workers=20)
    # for i_batch, sample_batched in enumerate(train_dataloader):
    #     print(i_batch, sample_batched['image'].size(),
    #           sample_batched['mask'].size())

    uavid_rgb_vals = {
        tuple([0, 0, 0]): 0,  # Background Clutter
        tuple([128, 0, 0]): 1,  # Building
        tuple([128, 64, 128]): 2,  # Road
        tuple([0, 128, 0]): 3,  # Tree
        tuple([128, 128, 0]): 4,  # Low Vegetation
        tuple([64, 0, 128]): 5,  # Moving Car
        tuple([192, 0, 192]): 6,  # Static Car
        tuple([192, 0, 192]): 7  # Human
    }

    dir_name = os.path.dirname(__file__)  # get the location of the root directory
    dataset = "uavid"
    dir_name = os.path.join(dir_name, '../..')
    dir_name = os.path.join(dir_name, 'data/segmentation-datasets')
    dir_name = os.path.join(dir_name, dataset)
    uav_train_set = RunwaysDataset(os.path.join(dir_name, 'train'), uavid_rgb_vals)
    uav_validation_set = RunwaysDataset(os.path.join(dir_name, 'validation'), uavid_rgb_vals)
    uav_train_dataloader = DataLoader(uav_train_set, batch_size=4, shuffle=False, num_workers=20)
    uav_validation_dataloader = DataLoader(uav_validation_set, batch_size=4, shuffle=False, num_workers=20)
    for i_batch, sample_batched in enumerate(uav_validation_dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['mask'].size())
        # img = sample_batched['image'][0].cpu().numpy()
        # # re-arrange order to HWC
        # img = np.clip(img, 0, 1)
        # plt.imshow(img)
        # plt.show()

    # cityscapes_rgb_vals = {
    #     tuple([0, 0, 0]): 0,  # Background Clutter
    #     tuple([111, 74, 0]): 1,  # Dynamic Object
    #     tuple([81, 0, 81]): 2,  # Ground
    #     tuple([128, 64, 128]): 3,  # Road
    #     tuple([244, 35, 232]): 4,  # Sidewalk
    #     tuple([250, 170, 160]): 5,  # Parking
    #     tuple([230, 150, 140]): 6,  # Rail-Track
    #     tuple([70, 70, 70]): 7,  # building
    #     tuple([102, 102, 156]): 8,  # wall
    #     tuple([190, 153, 153]): 9,  # fence
    #     tuple([180, 165, 180]): 10,  # guard-rail
    #     tuple([150, 100, 100]): 11,  # bridge
    #     tuple([150, 120, 90]): 12,  # tunnel
    #     tuple([153, 153, 153]): 13,  # pole
    #     tuple([250, 170, 30]): 14,  # traffic light
    #     tuple([220, 220, 0]): 15,  # traffic sign
    #     tuple([107, 142, 35]): 16,  # vegetation
    #     tuple([152, 251, 152]): 17,  # terrain
    #     tuple([70, 130, 180]): 18,  # sky
    #     tuple([220, 20, 60]): 19,  # person
    #     tuple([255, 0, 0]): 20,  # bike-rider
    #     tuple([0, 0, 142]): 21,  # car
    #     tuple([0, 0, 70]): 22,  # truck
    #     tuple([0, 60, 100]): 23,  # bus
    #     tuple([0, 0, 90]): 24,  # caravan
    #     tuple([0, 0, 110]): 25,  # trailer
    #     tuple([0, 80, 100]): 26,  # train
    #     tuple([0, 0, 230]): 27,  # motorcycle
    #     tuple([119, 11, 32]): 28,  # bicycle
    #     tuple([0, 0, 142]): 30  # license plate
    # }
    #
    # dir_name = os.path.dirname(__file__)  # get the location of the root directory
    # dataset = "cityscapes"
    # dir_name = os.path.join(dir_name, '../..')
    # dir_name = os.path.join(dir_name, 'data/segmentation-datasets')
    # dir_name = os.path.join(dir_name, dataset)
    # city_set = CityscapesDataset(dir_name, cityscapes_rgb_vals)
    # city_set_dataloader = DataLoader(city_set, batch_size=4, shuffle=False, num_workers=20)
    # for i_batch, sample_batched in enumerate(city_set_dataloader):
    #     print(i_batch, sample_batched['image'].size(),
    #           sample_batched['mask'].size())
    #     # img = sample_batched['image'][0].cpu().numpy()
    #     # # rearrage order to HWC
    #     # img = np.clip(img, 0, 1)
    #     # plt.imshow(img)
    #     # plt.show()
    # print('fin')

# for idx in range(len(validation_set)):
#     sample = validation_set[idx]
#     print(idx, sample['image'].shape, sample['mask'].shape)

