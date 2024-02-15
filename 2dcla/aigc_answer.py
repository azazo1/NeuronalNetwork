import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs


def loadSample() -> np.ndarray:
    return pickle.load(open("sample", 'rb'))
# 加载样本
sample = loadSample()
labels = sample[:, 2].astype('float')
data = sample[:, :2].astype('float')
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 构建神经网络模型
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.exFit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 评估模型
p = model.predict(X_test)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')
