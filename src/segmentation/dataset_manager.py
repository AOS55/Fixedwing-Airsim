import os
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


class RunwaysDataset(Dataset):
    """
    Prepare input and training data from the segmentation maps
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
        self.semantic_dir = semantic_dir
        self.image_dir = image_dir
        self.labels = labels

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
        mask_tensor = self.mask_preparation(mask_loc)
        image_tensor = self.image_preparation(image_loc)
        sample = {'image': image_tensor, 'mask': mask_tensor}
        return sample

    def mask_preparation(self, mask_name: str):
        """
        Takes an example of the pretrained mask and turns it into a tensor based on labels available.
        Based on the labels dictionary, remaps values of (rgb) triplets from the training set to use integer values.

        :param mask_name:
        :return: torch_mask: torch.Tensor of int values from dict, input_mask: PIL image type for resizing and outting
        """
        input_mask = Image.open(mask_name)
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
    def image_preparation(image_name: str):
        """
        Takes a regular png image and prepares it to be an input to the CNN

        :param image_name: the path + name to the input_image
        :return: mini_batch the tensor representation of images expected by the model
        """
        input_image = Image.open(image_name)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(input_image)
        return input_tensor


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

    dirname = os.path.dirname(__file__)  # get the location of the root directory
    dataset = "480-multicct"
    dirname = os.path.join(dirname, '../..')
    dirname = os.path.join(dirname, 'data/segmentation-datasets')
    dirname = os.path.join(dirname, dataset)
    category_rgb_vals = {
        tuple([0, 0, 0]): 0,
        tuple([78, 53, 104]): 1,
        tuple([155, 47, 90]): 2
    }
    data_set = RunwaysDataset(dirname, category_rgb_vals)
    split_data_set = split_dataset(data_set, 0.25)

    train_dataloader = DataLoader(split_data_set['train'], batch_size=4, shuffle=False, num_workers=10)
    test_dataloader = DataLoader(split_data_set['test'], batch_size=4, shuffle=False, num_workers=10)
    for i_batch, sample_batched in enumerate(train_dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['mask'].size())

# for idx in range(len(test_set)):
#     sample = test_set[idx]
#     print(idx, sample['image'].shape, sample['mask'].shape)

