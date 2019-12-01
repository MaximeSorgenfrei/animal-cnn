import cv2
import keras
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from random import shuffle
# import tensorflow as tf
import time

file_list = os.listdir("./")
keras_model_files = []
for file in file_list:
    if file.split(".")[-1] in ["h5","npy"]:
        print(file)
        keras_model_files.append(file)

# load model from file
keras_model_file_i_want_to_use = keras_model_files[-5]
print("Using following model file for predictions:\n{}".format(keras_model_file_i_want_to_use))
model = keras.models.load_model(keras_model_file_i_want_to_use)
model.summary()
# classes = ["ape", "bear", "bee", "beetle", "bird", "bos", "canine", "deer", "elephants", "feline", "frogs", "gekko", "golden moles", "hare", "human", "lemur", "loris", "none", "rodent", "salamander", "scorpions", "shark", "sheep", "snake", "spider", "squirrel", "turtle", "whale"]
# read directories, resize and label data
# Write some Text

# dict
with open(keras_model_files[-3],"r") as f:
    class_list = json.load(f)
    class_stats = pd.DataFrame(data={"classes":class_list})
    classes = class_stats["classes"].to_dict()
f.close()
print(classes)

font                   = cv2.FONT_HERSHEY_COMPLEX
bottomLeftCornerOfText = (50,50)
bottomLeftCornerOfText2 = (50,75)
fontScale              = 0.5
fontColor              = (255,255,255)
lineType               = 2

width, height = 50, 50
WebCam_cap = cv2.VideoCapture(0)

while True:
    # get frame
    ret, frame = WebCam_cap.read()
#     print(type(frame))
    try:
        # reduce frame to 50x50 pixles
        #     image = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_to_predict = np.reshape(image, (1, width, height, 1))
        # predict with NN
        pred = model.predict_classes(image_to_predict, verbose=0)
        pred_ = model.predict(image_to_predict, verbose=0)
        prediction = "{}: {} | {}: {}".format(classes[0], pred_[0][0], classes[1], pred_[0][1])
        if pred_[0][pred[0]] > 0.30:
            prediction_class = "Predicted class: {} [{:.2f}]".format(classes[pred[0]], pred_[0][pred[0]])
        else:
            prediction_class = "No significant prediction possible!"
        # print prediction and class to frame
        # cv2.putText(frame, prediction, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        cv2.putText(frame, prediction_class, bottomLeftCornerOfText2, font, fontScale, fontColor, lineType)
        #     ax[i].set_title("{}: {}-{} ({})".format(i, pred, classes[pred[0]], np.round(pred_, decimals=4)))
        # display resut
#         cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
#         cv2.imshow("Result", image)
    except Exception as e:
        print(e)
    else:
        cv2.namedWindow("WebCam", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("WebCam", frame)
    
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
        
WebCam_cap.release()
cv2.destroyAllWindows()
