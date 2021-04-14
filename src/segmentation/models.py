import segmentation_models_pytorch as smp
import torch
import torchvision.models as models


def get_model(model_name: str, device, num_classes: int) -> torch.nn:
    """
    Gets the model requested from the config file

    :param model_name: name of model used in the program
    :param device: name of device model is hosted on
    :param num_classes: number of classes in the model (including background 0,0,0)
    :return: the nn.model to be trained
    """
    model_dict = {'deeplabv3': torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101',
                                              pretrained=False, num_classes=num_classes).to(device).eval(),
                  'UnetPlusPlus': smp.UnetPlusPlus(encoder_name='resnet34', encoder_depth=5,
                                                   encoder_weights=None, classes=num_classes).to(device).eval(),
                  'deeplabv3plus': smp.DeepLabV3Plus(encoder_name='resnet101', encoder_depth=5,
                                                     encoder_weights=None, classes=num_classes).to(device).eval(),
                  'resnet50': models.segmentation.fcn_resnet50(pretrained=False, num_classes=num_classes).to(
                      device).eval(),
                  'lraspp_mobile': models.segmentation.lraspp_mobilenet_v3_large(pretrained=False,
                                                                                 num_classes=num_classes).to(
                      device).eval()}
    try:
        model = model_dict[model_name]
    except KeyError:
        print(f"KeyError, model_name is not valid allowable names are: {model_dict.keys()}")
        model = None
    return model


if __name__ == '__main__':
    print('Dont Run models as main')
