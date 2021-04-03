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

        rgb_vals = {
            'runway': runway_rgb_vals,
            'uav': uav_rgb_vals
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
