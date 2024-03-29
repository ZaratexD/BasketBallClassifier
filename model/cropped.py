# first iteration of classifier
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import os
import shutil

img = cv2.imread("model/test_images/k_2.jpg")

temp_path = "model/test_images/k_2.jpg"

# used to clean name of raw google images into folder model/dataset/names fixed


def rename_google_scrape(raw_images):
    fixed_folder_path = "model/dataset/names fixed"
    for player in os.scandir(raw_images):
        if player.is_dir():
            count = 0
            player_name = player.path.split("/")[-1:][0].split("-")[0].strip()
            new_dir = fixed_folder_path + "/" + player_name
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            for images in os.scandir(player):
                curr = images.path.split(".")[-1:][0]
                if curr == "jpg" or curr == "png":
                    new_img_path = new_dir + "/" + player_name + str(count) + "." + curr

                    count = count + 1
                    cv2.imwrite(new_img_path, cv2.imread(images.path))


raw_images = "model/dataset/raw google scrape"
rename_google_scrape(raw_images)


# Returns a cropped image if face == true and eyes >=2
def cropped_face(image_path):
    curr_image = cv2.imread(image_path)
    gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)  # why convert to gray?

    # review what these haarcascade do
    face_cascade = cv2.CascadeClassifier(
        "model/opencv/haarcascades/haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier("model/opencv/haarcascades/haarcascade_eye.xml")

    faces = face_cascade.detectMultiScale(
        gray, 1.3, 5
    )  # gray because less processing?, what do other params do

    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = curr_image[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color


path_to_data = "model/dataset/names fixed"
path_to_cr_data = "model/dataset/cropped/"
img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)


if os.path.exists(path_to_cr_data):
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)

cropped_image_dirs = []
celebrity_file_names_dict = {}
for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split("/")[-1]
    celebrity_file_names_dict[celebrity_name] = []
    for entry in os.scandir(img_dir):
        if entry.path[-3:] == "jpg":
            roi_color = cropped_face(entry.path)
            if roi_color is not None:
                cropped_folder = path_to_cr_data + celebrity_name
                if not os.path.exists(cropped_folder):
                    os.makedirs(cropped_folder)
                    cropped_image_dirs.append(cropped_folder)
                    print("Generating cropped images in folder: ", cropped_folder)
                cropped_file_name = celebrity_name + str(count) + ".png"
                cropped_file_path = cropped_folder + "/" + cropped_file_name
                cv2.imwrite(cropped_file_path, roi_color)
                celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
                count += 1
