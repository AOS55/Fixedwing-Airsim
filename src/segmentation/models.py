import torch
import segmentation_models_pytorch as smp
from config import default_config

device = torch.device(default_config['device'] if torch.cuda.is_available() else 'cpu')
print(f"Found device: {device}")
# device = torch.device('cpu')


def get_model(model_name: str) -> torch.nn:
    """
    Gets the model requested from the config file

    :param model_name:
    :return: the nn.model to be trained
    """
    model_dict = {'deeplabv3': torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101',
                                              pretrained=False, num_classes=3).to(device).eval(),
                  'UnetPlusPlus': smp.UnetPlusPlus(encoder_name='resnet34', encoder_depth=5,
                                                   encoder_weights=None, classes=3).to(device).eval(),
                  'deeplabv3plus': smp.DeepLabV3Plus(encoder_name='resnet101', encoder_depth=5,
                                                     encoder_weights=None, classes=3).to(device).eval()}
    try:
        model = model_dict[model_name]
    except KeyError:
        print(f"KeyError, model_name is not valid allowable names are: {model_dict.keys()}")
    return model


if __name__ == '__main__':
    model = get_model(default_config['model_name'])
    print(model)
