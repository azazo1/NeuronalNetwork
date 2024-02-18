import os
import pickle

import numpy as np
from keras import Sequential
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer

from nn_my_utils import readSingleImage, readImgTrainingData


def readModel(modelPath: str, labelBinPath: str):
    model = load_model(modelPath)  # type: Sequential
    with open(labelBinPath, "rb") as r:
        labelBin = pickle.load(r)  # type: LabelBinarizer
    return model, labelBin


def main():
    model, labelBin = readModel("output/simple_nn.model", "output/simple_nn_lb.pickle")
    prefix = r"D:\Temp"
    while True:
        img = readSingleImage(os.path.join(prefix, input("Img:")))
        img = img.astype('float') / 255
        get = model.predict(np.array([img]))
        print(get)


if __name__ == '__main__':
    main()
