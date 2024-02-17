from keras.initializers.initializers_v1 import TruncatedNormal
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from keras import regularizers
import nn_my_utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

# -- dataset --model --label-bin --plot
# 输入参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True, help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True, help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True, help="path to output accuracy/loss plot")
parse_args = ap.parse_args()
args = vars(parse_args)

print("[INFO] 开始读取数据")

# 拿到图像数据路径，方便后续读取
imagesBatch = nn_my_utils.readImgTrainingData(args['dataset'])
data = imagesBatch[b'data']  # type: np.ndarray
labels = np.array(imagesBatch[b'labels'], dtype="uint8")  # type:np.ndarray

# 选前三个分类 飞机 车 鸟类
data = data[labels < 3]
labels = labels[labels < 3]
data = data.astype('float') / 255.0

# 数据集划分
(trainX, testX, trainY, testY) = train_test_split(
    data,
    labels,
    test_size=0.25,
    random_state=42,
    shuffle=True
)

# 转换标签，one-hot格式
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# 网格模型结构：3072-512-256-3
model = Sequential()
# kernel_regularizer=regularizers.l2(0.01)
# keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
# initializers.random_normal
# #model.add(Dropout(0.8))
model.add(Dense(512, input_shape=(3072,), activation="relu",
                kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05),
                kernel_regularizer=regularizers.l2(0.02)))
# model.add(Dropout(0.4))
model.add(Dense(256, activation="relu",
                kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05),
                kernel_regularizer=regularizers.l2(0.02)))
# model.add(Dropout(0.4))
model.add(Dense(len(lb.classes_), activation="softmax",
                kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05),
                kernel_regularizer=regularizers.l2(0.02)))

# 初始化参数
INIT_LR = 0.003
EPOCHS = 5000

# 给定损失函数和评估方法
print("[INFO] 准备训练网络...")
opt = SGD(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# 训练网络模型
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=7500)

# 测试网络模型
print("[INFO] 正在评估模型")
predictions = model.predict(testX, batch_size=2500)
print(
    classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(i) for i in lb.classes_]))

# 当训练完成时，检测结果曲线
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history['loss'], label='train_loss')
plt.plot(N, H.history['val_loss'], label='val_loss')
plt.plot(N, H.history['accuracy'], label="accuracy")
plt.plot(N, H.history['val_accuracy'], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
# plt.ylim(0, 1)
plt.legend()
plt.savefig(args["plot"])

# 保存模型到本地
print("[INFO] 正在保存模型")
model.save(args['model'])
f = open(args['label_bin'], 'wb')
f.write(pickle.dumps(lb))
f.close()

plt.show()
