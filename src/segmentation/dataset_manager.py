import os
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import matplotlib.pyplot as plt

# class UAVDataset(RunwaysDataset):
#     """
#     Prepare input and training data from the iUAV dataset https://uavid.nl/
#     """
#
#     def __init__(self,
#                  root_dir: str,
#                  labels: dict,
#                  semantic_dir: str = "segmentation_masks",
#                  image_dir: str = "images"):
#         """
#         Args:
#         :param root_dir: Root directory of all dataset files
#         :param semantic_dir: path to directory containing all semantic images
#         :param image_dir: path containing all raw images
#         :param labels: dictionary containing the rgb triplets and labels of the image mask
#         """
#         self.root_dir = root_dir
#         self.semantic_dir = semantic_dir
#         self.image_dir = image_dir
#         self.labels = labels
#
#     def __len__(self):
#         """
#         Get the number of frames in the dataset
#         :return: simply the number of files in the semantic image directory
#         """
#         dirname = os.path.join(self.root_dir, self.semantic_dir)
#         return len([name for name in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, name))])
#
#     def __getitem__(self, idx):
#         """
#         Get samples from the dataset
#         :param idx: the index of the tensor requested
#         :return: 2 dicts of tensors image and mask for test and train samples
#         """


class RunwaysDataset(Dataset):
    """
    Prepare input and training data from the UE4 segmentation maps
    """

    def __init__(self,
                 root_dir: str,
                 labels: dict,
                 semantic_dir: str = "segmentation_masks",
                 image_dir: str = "images"):
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


class UAVDataset(RunwaysDataset):
    """
    Loads in a dataset with the same form as the RunwaysDataset but with more categories and different transforms
    |___test (root_dir)
    |   |_images
    |   |_segmentation_masks
    |
    |___train (root_dir)
    |   |_images
    |   |_segmentation_masks
    """
    def __init__(self, root_dir: str, labels: dict, crop_size: tuple = (480, 852)):
        super().__init__(root_dir, labels, )
        self.root_dir = root_dir
        self.labels = labels
        self.crop_size = crop_size

    def __getitem__(self, idx):
        mask_loc = os.path.join(self.root_dir, self.semantic_dir, str(idx) + ".png")
        image_loc = os.path.join(self.root_dir, self.image_dir, str(idx) + ".png")
        input_mask = Image.open(mask_loc)
        input_image = Image.open(image_loc)
        input_image, input_mask = self.apply_random_crop(input_image, input_mask)
        mask_tensor = self.mask_preparation(input_mask)
        image_tensor = self.image_preparation(input_image)
        sample = {'image': image_tensor, 'mask': mask_tensor}
        return sample

    def apply_random_crop(self, img: Image.Image, tgt: Image.Image,):
        """
        Apply a random crop equally to the target and the image

        :param img: a PIL Image of the RGB image
        :param tgt: a PIL Image of the target image
        :param scale: an HxW scale of the desired crop e.g. 480x852
        :return: the cropped img & tgt tensors
        """
        t = transforms.RandomResizedCrop(self.crop_size)
        state = torch.get_rng_state()
        img = t(img)
        torch.set_rng_state(state)
        tgt = t(tgt)
        return img, tgt


def split_dataset(dataset, split_size: float = 0.25):
    """
    Train test split

    :param dataset: the dataset to be split
    :param split_size: the test to train split
    """
    train_idx_list, test_idx_list = train_test_split(list(range(len(dataset) - 1)), test_size=split_size)
    datasets = {'train': Subset(dataset, train_idx_list), 'test': Subset(dataset, test_idx_list)}
    return datasets


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
    # test_dataloader = DataLoader(split_data_set['test'], batch_size=4, shuffle=False, num_workers=20)
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
    uav_train_set = UAVDataset(os.path.join(dir_name, 'train'), uavid_rgb_vals)
    uav_test_set = UAVDataset(os.path.join(dir_name, 'test'), uavid_rgb_vals)
    uav_train_dataloader = DataLoader(uav_train_set, batch_size=4, shuffle=False, num_workers=20)
    uav_test_dataloader = DataLoader(uav_test_set, batch_size=4, shuffle=False, num_workers=20)
    for i_batch, sample_batched in enumerate(uav_test_dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['mask'].size())
        # img = sample_batched['image'][0].cpu().numpy()
        # # rearrage order to HWC
        # img = np.clip(img, 0, 1)
        # plt.imshow(img)
        # plt.show()



# for idx in range(len(test_set)):
#     sample = test_set[idx]
#     print(idx, sample['image'].shape, sample['mask'].shape)

