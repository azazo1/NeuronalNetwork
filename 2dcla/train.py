import pickle

import numpy as np
from keras import Sequential, regularizers
from keras.initializers.initializers_v1 import TruncatedNormal
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split


def loadSample() -> np.ndarray:
    return pickle.load(open("sample", 'rb'))


def main():
    # 加载样本
    sample = loadSample()
    labels = sample[:, 2].astype('uint8')
    data = sample[:, :2].astype('float')
    # 划分训练集和测试集
    (trainX, testX, trainY, testY) = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42
    )
    # 构建模型并添加神经网络
    model = Sequential()
    model.add(Dense(32, input_shape=(2,), activation='tanh',
                    # kernel_initializer=TruncatedNormal(seed=42),
                    # kernel_regularizer=regularizers.l2(0.02)
                    ))
    model.add(Dense(1, activation='tanh',
                    # kernel_initializer=TruncatedNormal(seed=42),
                    # kernel_regularizer=regularizers.l2(0.02)
                    ))  # 这里改为softmax后准确率根本不提升
    # 训练
    model.compile(optimizer=Adam(lr=0.006), loss="binary_crossentropy", metrics=['accuracy'])
    acc = 0
    H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=1000, batch_size=32 * 24)
    # 测试
    # P = model.predict(testX)
    loss, acc = model.evaluate(testX, testY)
    model.save("trained.model")


if __name__ == '__main__':
    main()
