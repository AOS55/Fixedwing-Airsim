import torch
from torch.utils import data
import cv2
import os


class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = cv2.imread(input_ID), cv2.imread(target_ID)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        print(x, y)
        return x, y


dirname = os.path.dirname(__file__)
image_loc = dirname + "/default_dataset/images/"
target_loc = dirname + "/default_dataset/segmentation_masks/"

inputs = [image_loc + "20210317-110751.png", image_loc + "20210317-110753.png"]
targets = [target_loc + "20210317-110750.png", target_loc + "20210317-110752.png"]

training_dataset = SegmentationDataSet(inputs=inputs, targets=targets, transform=None)
training_dataloader = data.DataLoader(dataset=training_dataset,
                                      batch_size=2,
                                      shuffle=True)
# print(training_dataloader)

x, y = next(iter(training_dataloader))

print(f"x = shape: {x.shape}; type: {x.dtype}")
print(f"x = min: {x.min()}; max: {x.max()}")
print(f"Y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}")
