# import segmentation_models_pytorch as smp
import torch
# import torchvision.models as models


def get_model(model_name: str, device, num_classes: int, pretrained: bool = False) -> torch.nn:
    """
    Gets the model requested from the config file

    :param model_name: name of model used in the program
    :param device: name of device model is hosted on
    :param num_classes: number of classes in the model (including background 0,0,0)
    :param pretrained: boolean of starting on a pretrained model
    :return: the nn.model to be trained
    """
    model_dict = {'fcn_resnet50': torch.hub.load('pytorch/vision', 'fcn_resnet50', pretrained=pretrained,
                                                 num_classes=num_classes).to(device).eval(),
                  'fcn_resnet101': torch.hub.load('pytorch/vision', 'fcn_resnet101', pretrained=pretrained,
                                                  num_classes=num_classes).to(device).eval(),
                  'deeplabv3_resnet50': torch.hub.load('pytorch/vision', 'deeplabv3_resnet50', pretrained=pretrained,
                                                       num_classes=num_classes).to(device).eval(),
                  'deeplabv3_resnet101': torch.hub.load('pytorch/vision', 'deeplabv3_resnet101',
                                                        pretrained=pretrained, num_classes=num_classes).to(
                      device).eval(),
                  'deeplabv3_mobilenet_v3_large': torch.hub.load('pytorch/vision', 'deeplabv3_mobilenet_v3_large',
                                                                 pretrained=pretrained, num_classes=num_classes).to(
                      device).eval(),
                  'lraspp_mobilenet_v3_large': torch.hub.load('pytorch/vision', 'lraspp_mobilenet_v3_large',
                                                              pretrained=pretrained, num_classes=num_classes).to(
                      device).eval()
                  }
    try:
        model = model_dict[model_name]
    except KeyError:
        print(f"KeyError, model_name is not valid allowable names are: {model_dict.keys()}")
        model = None
    return model


if __name__ == '__main__':
    print('Dont Run models as main')
