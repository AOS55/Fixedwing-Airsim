import argparse


class NetworkConfig:
    """
    Setup a custom datastructure for the Convolutional Neural network with shell script
    """

    def __init__(self,
                 epochs: int = 20,
                 learning_rate: float = 1e-3,
                 batch_size: int = 1,
                 model_name: str = 'resnet18',
                 device: str = 'cuda',
                 dataset: str = '480-multicct',
                 class_name: str = 'runway',
                 run_name: str = '480-multicct',
                 num_workers: int = 4,
                 image_height: int = 480,
                 image_width: int = 852
                 ):
        """
        Run the parser and change the config file used
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_name = model_name
        self.device = device
        self.dataset = dataset
        self.class_name = class_name
        self.run_name = run_name
        self.num_workers = num_workers
        self.image_height = image_height
        self.image_width = image_width
        self.parser_args = self.parser_func()
        self.build_config_file()
        self.classes = self.rgb_classes()

    @staticmethod
    def parser_func() -> argparse.ArgumentParser.parse_args:
        """
        Setup a config file based on user input

        :return: a parser object of args
        """
        parser = argparse.ArgumentParser(description='CNN configuration')
        parser.add_argument('--epochs', type=int, help='number of times to run the dataset through the nn')
        parser.add_argument('--learning_rate', type=float, help='how much to update the weights and biases by [0-1]')
        parser.add_argument('--batch_size', type=int, help='the number of samples to propogate through the network on '
                                                           'each pass')
        parser.add_argument('--model_name', type=str, help='the name of the nn model used')
        parser.add_argument('--device', type=str, help='the name of the device type to train the nn on'
                                                       ' either CUDA or CPU')
        parser.add_argument('--dataset', type=str, help='the name of the dataset dir to train the nn')
        parser.add_argument('--class_name', type=str, help='the name of the class dict contained used in the nn seg '
                                                           'map (runway_rgb_vals by default)')
        parser.add_argument('--run_name', type=str, help='the name of the directory to store the results of the nn')
        parser.add_argument('--num_workers', type=int, help='the number of workers to use for batch loading')
        parser.add_argument('--image_height', type=int, help='height of RGB images if cropping images')
        parser.add_argument('--image_width', type=int, help='the width of RGB images if cropping images')
        args = parser.parse_args()
        return args

    def build_config_file(self):
        """
        Build a config dictionary to be used for image segmentation

        :return: the config dictionary modified by config
        """
        if self.parser_args.epochs:
            self.epochs = self.parser_args.epochs
        if self.parser_args.learning_rate:
            self.learning_rate = self.parser_args.learning_rate
        if self.parser_args.batch_size:
            self.batch_size = self.parser_args.batch_size
        if self.parser_args.model_name:
            self.model_name = self.parser_args.model_name
        if self.parser_args.device:
            self.device = self.parser_args.device
        if self.parser_args.dataset:
            self.dataset = self.parser_args.dataset
        if self.parser_args.class_name:
            self.class_name = self.parser_args.class_name
        if self.parser_args.run_name:
            self.run_name = self.parser_args.run_name
        else:
            self.run_name = 'mn' + self.model_name + 'eps' + str(self.epochs) + 'lr' + str(self.learning_rate) + 'bs' \
                            + str(self.batch_size)
        if self.parser_args.num_workers:
            self.num_workers = self.parser_args.num_workers
        if self.parser_args.image_height:
            self.image_height = self.parser_args.image_height
        if self.parser_args.image_width:
            self.image_width = self.parser_args.image_width

    def rgb_classes(self) -> dict:
        """
        Function to map the input string for the expected type of dict to the dict itself

        :return: dictionary of rgb_values
        """

        # Runway dataset colour map
        runway_rgb_vals = {
            tuple([0, 0, 0]): 0,  # Sky
            tuple([78, 53, 104]): 1,  # Runway
            tuple([155, 47, 90]): 2  # Ground
        }

        # UAVid dataset colour map
        uav_rgb_vals = {
            tuple([0, 0, 0]): 0,  # Background Clutter
            tuple([128, 0, 0]): 1,  # Building
            tuple([128, 64, 128]): 2,  # Road
            tuple([0, 128, 0]): 3,  # Tree
            tuple([128, 128, 0]): 4,  # Low Vegetation
            tuple([64, 0, 128]): 5,  # Moving Car
            tuple([192, 0, 192]): 6,  # Static Car
            tuple([192, 0, 192]): 7  # Human
        }

        cityscapes_rgb_vals = {
            tuple([0, 0, 0]): 0,  # Background Clutter
            tuple([111, 74, 0]): 1,  # Dynamic Object
            tuple([81, 0, 81]): 2,  # Ground
            tuple([128, 64, 128]): 3,  # Road
            tuple([244, 35, 232]): 4,  # Sidewalk
            tuple([250, 170, 160]): 5,  # Parking
            tuple([230, 150, 140]): 6,  # Rail-Track
            tuple([70, 70, 70]): 7,  # building
            tuple([102, 102, 156]): 8,  # wall
            tuple([190, 153, 153]): 9,  # fence
            tuple([180, 165, 180]): 10,  # guard-rail
            tuple([150, 100, 100]): 11,  # bridge
            tuple([150, 120, 90]): 12,  # tunnel
            tuple([153, 153, 153]): 13,  # pole
            tuple([250, 170, 30]): 14,  # traffic light
            tuple([220, 220, 0]): 15,  # traffic sign
            tuple([107, 142, 35]): 16,  # vegetation
            tuple([152, 251, 152]): 17,  # terrain
            tuple([70, 130, 180]): 18,  # sky
            tuple([220, 20, 60]): 19,  # person
            tuple([255, 0, 0]): 20,  # bike-rider
            tuple([0, 0, 142]): 21,  # car
            tuple([0, 0, 70]): 22,  # truck
            tuple([0, 60, 100]): 23,  # bus
            tuple([0, 0, 90]): 24,  # caravan
            tuple([0, 0, 110]): 25,  # trailer
            tuple([0, 80, 100]): 26,  # train
            tuple([0, 0, 230]): 27,  # motorcycle
            tuple([119, 11, 32]): 28,  # bicycle
            tuple([0, 0, 142]): 30  # license plate
        }

        cityscapes_labels = {
            0: (0, 0, 0),  # unlabeled
            1: (0, 0, 0),  # ego vehicle
            2: (0, 0, 0),  # rectification border
            3: (0, 0, 0),  # out of bounds
            4: (0, 0, 0),  # static
            5: (111, 74, 0),  # dynamic
            6: (81, 0, 81),  # ground
            7: (128, 64, 128),  # road
            8: (244, 35, 232),  # sidewalk
            9: (250, 170, 160),  # parking
            10: (230, 150, 140),  # rail track
            11: (70, 70, 70),  # building
            12: (102, 102, 156),  # wall
            13: (190, 153, 153),  # fence
            14: (180, 165, 180),  # guard rail
            15: (150, 100, 100),  # bridge
            16: (150, 120, 90),  # tunnel
            17: (153, 153, 153),  # pole
            18: (153, 153, 153),  # polegroup
            19: (250, 170, 30),  # traffic light
            20: (220, 220, 0),  # traffic sign
            21: (107, 142, 35),  # vegetation
            22: (152, 251, 152),  # terrain
            23: (70, 130, 180),  # sky
            24: (220, 20, 60),  # person
            25: (255, 0, 0),  # rider
            26: (0, 0, 142),  # car
            27: (0, 0, 70),  # truck
            28: (0, 60, 100),  # bus
            29: (0, 0, 90),  # caravan
            30: (0, 0, 110),  # trailer
            31: (0, 80, 100),  # train
            32: (0, 0, 230),  # motorcycle
            33: (119, 11, 32),  # bicycle
            34: (0, 0, 142)  # license plate
        }

        rgb_vals = {
            'runway': runway_rgb_vals,
            'uav': uav_rgb_vals,
            'cityscapes': cityscapes_labels
        }

        return rgb_vals[self.class_name]


runway_rgb_cats = {
    (0, 0, 0): "sky",
    (78, 53, 104): "runway",
    (155, 47, 90): "ground"
}

if __name__ == "__main__":
    nn = NetworkConfig()
    print(nn.epochs)
