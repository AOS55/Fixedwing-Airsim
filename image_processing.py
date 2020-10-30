import airsim
import numpy as np
import cv2 as cv


# def get_images(simulator):
#     """Get image from airsim"""
#     image_response = simulator.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
#     image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
#     image_rgb = image1d.reshape(image_response.height, image_response.width, 3)
#     return image_rgb[78:144, 27:227, 0:2].astype(float)


def show_webcam():
    vid = cv.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv.destroyAllWindows()


# show_webcam()
