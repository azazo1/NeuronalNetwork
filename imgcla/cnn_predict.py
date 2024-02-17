import traceback

import cv2
import numpy as np

from cnn_train import SimpleCNNModel, extractImagesNLabels, loadDataset

model = SimpleCNNModel()
model.load("CNNTrainedModel")
labelString = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
while True:
    try:
        imgPath = input("picPath:").strip("'\"")
        img = cv2.imread(imgPath)
        img = cv2.resize(img, (32, 32))
        imgA = img.astype('float') / 255
        predict = model.predict(np.array([imgA]))
        result = predict.argmax()
        text = f"{labelString[result]} {predict.flatten()[result]:.2%}"
        print(text)
        imgScaledBack = cv2.resize(imgA, (500, 500), interpolation=cv2.INTER_NEAREST)
        texted = cv2.putText(imgScaledBack, text, (3, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0))
        cv2.imshow(text, texted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        traceback.print_exc()
