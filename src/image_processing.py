import airsim
import numpy as np
# import cv2 as cv
import torch
from src.jsbsim_simulator import Simulation
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


class AirSimImages:
    """
    A class to get and process images returned from cameras within the AirSim graphical simulation

    ...

    Attributes:
    ----------
    sim : Simulation object
        represents the combined jsbsim and airsim instance

    """
    def __init__(self, sim: Simulation):
        self.sim = sim

    def get_np_image(self, image_type) -> np.array:
        """
        Gets images from camera '0' as a numpy array

        :type: the type of image used can be of the following:
            - airsim.ImageType.DisparityNormalized
            - airsim.ImageType.Segmentation
            - airsim.ImageType.Infrared
            - airsim.ImageType.Scene
            - airsim.ImageType.DepthPerspective
            - airsim.ImageType.DepthPlanner
            - airsim.ImageType.DepthVis
            - airsim.ImageType.SurfaceNormals
        :return: image_rgb numpy array of with 4 channels of image_type=type
        """
        image_responses = self.sim.client.simGetImages([airsim.ImageRequest('0',
                                                                            image_type,
                                                                            False,
                                                                            False)])
        image_response = image_responses[0]
        # Get a numpy array from the image_response
        image_1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
        # reshape array to a 3 channel image array 3 x H x W
        image_rgb = image_1d.reshape(3, image_response.height, image_response.width)
        return image_rgb

    def get_png_image(self):
        """
        Gets images from camera '0' as a png_image

        :type: the type of image used can be of the following:
            - airsim.ImageType.DisparityNormalized
            - airsim.ImageType.Segmentation
            - airsim.ImageType.Infrared
            - airsim.ImageType.Scene
            - airsim.ImageType.DepthPerspective
            - airsim.ImageType.DepthPlanner
            - airsim.ImageType.DepthVis
            - airsim.ImageType.SurfaceNormals
        :return: simple png image
        """
        image = self.sim.client.simGetImage('0', airsim.ImageType.Scene)
        return image


class SemanticImageSegmentation(AirSimImages):
    """
    Classify segments of an image based on the content observed within each segment
    """

    def __init__(self, sim: Simulation) -> None:
        super().__init__(sim)  # access all image methods contained in AirSimImages
        self.model = self.load_model()
        self.input_image = self.get_np_image(image_type=airsim.ImageType.Scene)
        self.input_batch = self.set_input_batch()

    @staticmethod
    def load_model():
        """
        Load a semantic segmentation model from torch hub

        :return: model of type class 'torchvision.models.segmentation.deeplabv3.DeepLabV3', a callable pytorch CNN
        """
        model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
        model.eval()  # run model in evaluation (validation) mode, NOT training
        return model

    def set_input_batch(self):
        """
        Set an input batch to use in the model

        :return: self.input_batch a minibatch expected as an input to the CNN seg-model
        """
        # input_image = Image.open(self.input_image)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        normalized_image = self.input_image
        input_tensor = preprocess(normalized_image)
        self.input_batch = input_tensor.unsqueeze(0)
        return self.input_batch

    def update_model(self):
        """
        Update and run the CNN segmentation model

        :return: output_predictions, the segmented image based on the model
        """
        self.set_input_batch()  # update input batch to current image
        if torch.cuda.is_available():
            self.input_batch = self.input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(self.input_batch)['out'][0]
        output_predictions = output.argmax(0) # returns [0] tensor at the moment?
        return output_predictions

    def get_segmented_image_plot(self):
        """
        Get a coloured plot where the image is segmented and colours assigned based on the class present in each
        part of the image
        """
        output_predictions = self.update_model()
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colours = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colours = (colours % 255).numpy().astype('uint8')

        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize((1920, 1080))
        r.putpalette(colours)

        plt.imshow(r)
        plt.show()


# def show_webcam():
#     vid = cv.VideoCapture(0)
#     while True:
#         ret, frame = vid.read()
#         cv.imshow('frame', frame)
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
#     vid.release()
#     cv.destroyAllWindows()
