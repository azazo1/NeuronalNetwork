import os.path
import pickle
from typing import List, Tuple, Union

import numpy as np
from keras import Sequential, regularizers
from keras.initializers.initializers_v1 import TruncatedNormal
from keras.layers import MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.losses import CategoricalCrossentropy
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.ops.variable_scope import variable_scope


# Conv 就是 Conv1D

class SimpleCNNModel:
    def __init__(self):
        self.model = Sequential()
        self.lb = LabelBinarizer()

    def build11(self):
        self.model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3), padding="same", activation="relu",
                              kernel_initializer=TruncatedNormal()))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), padding="same", activation="relu",
                              kernel_initializer=TruncatedNormal()))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01),
                              kernel_initializer=TruncatedNormal()))
        self.model.add(Conv2D(256, (3, 3), padding="same", activation="relu",
                              kernel_initializer=TruncatedNormal()))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(512, (3, 3), padding="same", activation="relu",
                              kernel_initializer=TruncatedNormal()))
        self.model.add(Conv2D(512, (3, 3), padding="same", activation="relu",
                              kernel_regularizer=regularizers.l2(0.01),
                              kernel_initializer=TruncatedNormal()))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(512, (3, 3), padding="same", activation="relu",
                              kernel_initializer=TruncatedNormal()))
        self.model.add(Conv2D(512, (3, 3), padding="same", activation="relu",
                              kernel_initializer=TruncatedNormal()))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(4096, activation="relu",
                             kernel_regularizer=regularizers.l2(0.01),
                             kernel_initializer=TruncatedNormal()))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation="relu",
                             kernel_regularizer=regularizers.l2(0.01),
                             kernel_initializer=TruncatedNormal()))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.lb.classes_), activation="softmax",
                             kernel_regularizer=regularizers.l2(0.01),
                             kernel_initializer=TruncatedNormal()))

    def build19(self):
        self.model.add(Conv2D(64, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal(),
                              input_shape=(32, 32, 3)))
        self.model.add(Conv2D(64, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(BatchNormalization())

        self.model.add(Conv2D(128, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(Conv2D(128, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(BatchNormalization())

        self.model.add(Conv2D(256, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(Conv2D(256, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(Conv2D(256, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(Conv2D(256, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(BatchNormalization())

        self.model.add(Conv2D(512, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(Conv2D(512, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(Conv2D(512, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(Conv2D(512, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(BatchNormalization())

        self.model.add(Conv2D(512, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(Conv2D(512, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(Conv2D(512, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(Conv2D(512, (3, 3), (1, 1), "same", activation="relu", kernel_initializer=TruncatedNormal()))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(4096, "relu", kernel_initializer=TruncatedNormal()))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, "relu", kernel_initializer=TruncatedNormal()))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.lb.classes_), "softmax", kernel_initializer=TruncatedNormal()))

    def fitLabels(self, rawLabel: np.ndarray):
        self.lb.fit(rawLabel)

    def transformLabels(self, *rawLabelBatches: np.ndarray):
        transformedLabelBatches = []
        for rawLabels in rawLabelBatches:
            transformedLabelBatches.append(self.lb.transform(rawLabels))
        return transformedLabelBatches

    def train(self, trainData, labels, epochs=100, batchSize=32, learningRate=0.05, testData=None, testLabels=None):
        self.model.compile(optimizer=Adam(learning_rate=learningRate),
                           loss=CategoricalCrossentropy(),
                           metrics=['accuracy'])
        return self.model.fit(trainData, labels,
                              validation_data=(
                                  testData, testLabels
                              ) if testData is not None and testLabels is not None else None,
                              epochs=epochs, batch_size=batchSize, use_multiprocessing=False)

    def evaluate(self, testData, testLabels, batchSize=32):
        return self.model.evaluate(testData, testLabels, batchSize)

    def predict(self, data: Union[np.ndarray, list]):
        return self.model.predict(data)

    def save(self, path="CNNTrainedModel"):
        self.model.save(os.path.join(path, "model"))
        with open(os.path.join(path, "LabelBinarizer.lb"), 'wb') as w:
            pickle.dump(self.lb, w)

    def load(self, path="CNNTrainedModel"):
        self.model = load_model(os.path.join(path, "model"))
        with open(os.path.join(path, "LabelBinarizer.lb"), 'rb') as r:
            self.lb = pickle.load(r)


def loadDataset(path: str):
    with open(path, "rb") as r:
        return pickle.load(r, encoding='bytes')


def extractImagesNLabels(*datasets: np.ndarray):
    """
    batch文件加载出来是个字典，有以下的键值
    b"batch_label" -> byte_string
    b"labels" -> list[int]
    b"data" -> ndarray: (10000, 3072)
    b"filenames" -> list[byte_string]

    此函数提取数据和标签的同时：
    将原本结构为1024*red+1024*blue+1024*green的图片结构转换为shape=(32,32,3)的图片结构
    """
    data = []
    rawLabels = []
    for dataset in datasets:
        rawLabels.extend(dataset[b'labels'])
        data.extend(list(dataset[b'data']))
    transformedData = []
    for i, singleImgData in enumerate(data):
        print(f"\rTotal:{len(data)}, Now:{i}, {i / len(data):.2%}", end='')
        # transformedImgData = np.zeros((3072,), dtype='float')
        # for i in range(1024):
        #     transformedImgData[3 * i] = singleImgData[i]
        #     transformedImgData[3 * i + 1] = singleImgData[i + 1024]
        #     transformedImgData[3 * i + 2] = singleImgData[i + 2048]
        # transformedImgData = transformedImgData.reshape((32, 32, 3))

        red = singleImgData[:1024].reshape(32, 32)
        green = singleImgData[1024:2048].reshape(32, 32)
        blue = singleImgData[2048:].reshape(32, 32)
        transformedImgData = np.dstack((red, green, blue)).astype("float") / 255

        transformedData.append(transformedImgData)
    print(f"\rTotal:{len(data)}, Now: {len(data)}, 100.00%", end="\n")
    return np.array(transformedData, 'float'), np.array(rawLabels, dtype='uint8')


def train():
    datasetPaths = ['./dataset/data_batch_1',
                    './dataset/data_batch_2',
                    './dataset/data_batch_3',
                    './dataset/data_batch_4',
                    './dataset/data_batch_5']
    testDatasetPath = './dataset/test_batch'
    epochsPerLargeBatch = 100

    print("[INFO] Creating model")
    model = SimpleCNNModel()
    model.fitLabels(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    model.build11()
    histories = []
    for seq, datasetPath in enumerate(datasetPaths):
        print(f"[INFO] Loading training dataset {seq}")
        dataset = loadDataset(datasetPath)

        print("[INFO] Extracting image data and labels")
        imgData, rawLabels = extractImagesNLabels(dataset)
        (labels,) = model.transformLabels(rawLabels)
        trainData, testData, trainLabels, testLabels = train_test_split(
            imgData,
            labels,
            test_size=0.1
        )

        print("[INFO] Start training")
        H = model.train(trainData, trainLabels, epochsPerLargeBatch, 128, 0.0001, testData, testLabels)
        histories.append(H)

    print("[INFO] Loading test dataset")
    testDataset = loadDataset(testDatasetPath)
    testImgData, testLabels = extractImagesNLabels(testDataset)
    testLabels = model.transformLabels(testLabels)
    print("[INFO] Evaluating")
    loss, accuracy = model.evaluate(testImgData, testLabels)
    print(f"[Result] loss={loss:.2f}, accuracy={accuracy:.2f}")
    print("[INFO] Saving")
    model.save("CNNTrainedModel")

    print("[INFO] Showing")
    plt.style.use("ggplot")
    plt.figure()
    for i in range(len(datasetPaths)):
        N = np.arange(epochsPerLargeBatch * i, epochsPerLargeBatch * (i + 1))
        plt.plot(N, histories[i].history['loss'], label=f"loss_{i}")
        plt.plot(N, histories[i].history['accuracy'], label=f"accuracy_{i}")
        plt.plot(N, histories[i].history['val_loss'], label=f"val_loss_{i}")
        plt.plot(N, histories[i].history['val_accuracy'], label=f"val_accuracy_{i}")
    plt.legend()
    plt.savefig("CNNTraining.jpg")
    plt.show()


def predict():
    pass


def main():
    train()


if __name__ == '__main__':
    main()
