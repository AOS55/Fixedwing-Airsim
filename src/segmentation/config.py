import argparse


class NetworkConfig:
    """
    Setup a custom datastructure for the Convolutional Neural network with shell script
    """

    def __init__(self,
                 epochs: int = 20,
                 learning_rate: float = 1e-3,
                 batch_size: int = 1,
                 model_name: str = 'deeplabv3',
                 device: str = 'cuda',
                 dataset: str = 'tom-showcase',
                 classes: str = 3,
                 run_name: str = 'tom-showcase'
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
        self.classes = classes
        self.run_name = run_name
        self.parser_args = self.parser_func()
        self.build_config_file()

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
        parser.add_argument('--device', type=str, help='the name of the device type to train the nn on either CUDA or CPU')
        parser.add_argument('--dataset', type=str, help='the name of the dataset dir to train the nn')
        parser.add_argument('--classes', type=int, help='the number of classes contained in the nn input (3 by default)')
        parser.add_argument('--run_name', type=str, help='the name of the directory to store the results of the nn')
        args = parser.parse_args()
        return args

    def build_config_file(self):
        """
        Build a config dictionary to be used for image segmentation

        :param parser_args: a parser argument contianing all the arguments expected by the program
        :param config: The configuration dictionary containin default key:value pairs
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
        if self.parser_args.classes:
            self.classes = self.parser_args.classes
        if self.parser_args.run_name:
            self.run_name = self.parser_args.run_name


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

if __name__ == "__main__":
    nn = NetworkConfig()
    print(nn.epochs)
