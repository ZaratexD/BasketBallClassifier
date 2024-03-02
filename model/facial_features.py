import numpy as np
import pywt
import cv2
import os

cropped_img_path = "model/dataset/cropped"

# key: name of player, value: paths to their cropped face
cropped_img_dirs = {}

for player in os.scandir(cropped_img_path):
    player_name = player.path.split("/")[-1]
    cropped_img_dirs[player_name] = []
    for img in os.scandir(player.path):
        cropped_img_dirs[player_name].append(img.path)


# need to connect the /cropped images into here


# def w2d(img, mode="haar", level=1):
#     imArray = img
#     # Datatype conversions
#     # convert to grayscale
#     imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
#     # convert to float
#     imArray = np.float32(imArray)
#     imArray /= 255
#     # compute coefficients
#     coeffs = pywt.wavedec2(imArray, mode, level=level)

#     # Process Coefficients
#     coeffs_H = list(coeffs)
#     coeffs_H[0] *= 0

#     # reconstruction
#     imArray_H = pywt.waverec2(coeffs_H, mode)
#     imArray_H *= 255
#     imArray_H = np.uint8(imArray_H)

#     return imArray_H
