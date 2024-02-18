import sys

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Resampling

sys.path.append("../..")
from imgcla.cifar.cnn_train import SimpleCNNModel


def convert4BytesToInt(bytes_: bytes):
    num = 0
    for byte in bytes_:
        num *= 256
        num += byte
    return num


def loadLabelFile(filename: str):
    with open(filename, 'rb') as r:
        magicNumber = r.read(4)
        assert magicNumber == b'\x00\x00\x08\x01'  # 表明是label文件
        num = convert4BytesToInt(r.read(4))
        arr = np.zeros((num,))
        arr[:] = list(r.read())
        return arr.astype('uint8')


def loadImageFile(filename: str):
    with open(filename, "rb") as r:
        magicNumber = r.read(4)
        assert magicNumber == b'\x00\x00\x08\x03'  # 表明是image文件
        num = convert4BytesToInt(r.read(4))
        rows = convert4BytesToInt(r.read(4))
        cols = convert4BytesToInt(r.read(4))

        arr = np.zeros((num, 32, 32, 1))
        for n in range(num):
            read = r.read(rows * cols)
            singleImg = np.array(list(read)).reshape((28, 28))
            img = Image.fromarray(singleImg)
            img = img.resize((32, 32), resample=Resampling.NEAREST) # 若不加这个重采样方法，会导致准确率根本提不上去
            arr[n] = np.asarray(img).reshape((32, 32, 1))
        return arr.astype('float16') / 255


labels = loadLabelFile("./dataset/train-labels.idx1-ubyte")
testLabels = loadLabelFile("./dataset/t10k-labels.idx1-ubyte")
images = loadImageFile("./dataset/train-images.idx3-ubyte")
testImages = loadImageFile("./dataset/t10k-images.idx3-ubyte")

model = SimpleCNNModel()
model.load("VGGTrainedMnistModel")
# model.fitLabels(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
labels, testLabels = model.transformLabels(labels, testLabels)

# model.buildVGG11((32, 32, 1))
# H = model.train(images, labels, 100, 64, 0.001, testData=testImages, testLabels=testLabels)
print(model.evaluate(testImages, testLabels))
model.save("VGGTrainedMnistModel")
