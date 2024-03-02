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


def w2d(img, mode="haar", level=1):
    imArray = img
    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H


class_dict = {}
count = 0
for celebrity_name in cropped_img_dirs.keys():
    class_dict[celebrity_name] = count
    count = count + 1

x, y = [], []
for celebrity_name, training_files in cropped_img_dirs.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, "db1", 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack(
            (
                scalled_raw_img.reshape(32 * 32 * 3, 1),
                scalled_img_har.reshape(32 * 32, 1),
            )
        )
        x.append(combined_img)
        y.append(class_dict[celebrity_name])

x = np.array(x).reshape(len(x), 4096).astype(float)
