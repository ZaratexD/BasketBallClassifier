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

X, y = [], []

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
        X.append(combined_img)
        y.append(class_dict[celebrity_name])

X = np.array(X).reshape(len(X), 4096).astype(float)


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=10))])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))


print(class_dict)
print(classification_report(y_test, pipe.predict(X_test)))
print(len(X_test))

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

model_params = {
    "svm": {
        "model": svm.SVC(gamma="auto", probability=True),
        "params": {"svc__C": [1, 10, 100, 1000], "svc__kernel": ["rbf", "linear"]},
    },
    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {"randomforestclassifier__n_estimators": [1, 5, 10]},
    },
    "logistic_regression": {
        "model": LogisticRegression(solver="liblinear", multi_class="auto"),
        "params": {"logisticregression__C": [1, 5, 10]},
    },
}

scores = []
best_estimators = {}
import pandas as pd

for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp["model"])  # what is sklearn pipeline
    clf = GridSearchCV(pipe, mp["params"], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append(
        {"model": algo, "best_score": clf.best_score_, "best_params": clf.best_params_}
    )
    best_estimators[algo] = clf.best_estimator_

df = pd.DataFrame(scores, columns=["model", "best_score", "best_params"])

best_clf = best_estimators["svm"]

import joblib

# Save the model as a pickle in a file
# TODO save this to server automatically
joblib.dump(best_clf, "saved_model.pkl")

import json

with open("class_dictionary.json", "w") as f:
    f.write(json.dumps(class_dict))
