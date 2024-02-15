import os
import pickle

import cv2


def readImgTrainingData(dataset_path: str, batch_label=1):
    with open(os.path.join(dataset_path, f'data_batch_{batch_label}'), 'rb') as r:
        get_dict = pickle.load(r, encoding='bytes')
        # 用于显示图片
        # img_data = get_dict[b'data'][17]
        # img = Image.new("RGB", (32, 32), (255, 255, 255))
        # for j in range(1024):
        #     img.putpixel((j % 32, j // 32), (img_data[j], img_data[j + 1024], img_data[j + 2048]))
        # img.show("hello")
    return get_dict


def listImages(path="."):
    rst = []
    for d, sd, sf in os.walk(path):
        for f in sf:
            if f.endswith(".png") or f.endswith("jpg") or f.endswith("webp"):
                rst.append(os.path.join(d, f))
    return rst


def readSingleImage(imgPath: str):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = img.flatten('F')
    return img


if __name__ == '__main__':
    print(readImgTrainingData('dataset'))
