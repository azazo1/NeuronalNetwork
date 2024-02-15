"""
模型训练结果可视化

不断添加额外训练
背景中蓝色表示绿点预测区，黑色表示红点预测区
显示预测区更细致（step较小）会导致预测时间较长而导致额外训练受影响
左上角显示当前的loss和accuracy

可操作的：
左键添加新红点
右键添加新绿点
lshift删除最后一个新添加的点
鼠标滚轮向上使预测区显示更加细致，反之更加粗糙
回车保存额外训练后的模型
"""
import math
import threading
from typing import Tuple, Optional
import pickle
import numpy as np
import pygame
from keras import Sequential
from keras.models import load_model


def loadSample(samplePath: str) -> np.ndarray:
    return pickle.load(open(samplePath, 'rb'))


def getPredictionGraph(model: Sequential, size: int, step: int) -> Tuple[list, np.ndarray]:
    points = []
    for x in range(0, size, step):
        for y in range(0, size, step):
            points.append((x, y))
    arr = np.array(points) / size
    return points, model.predict(arr)


class Display:
    def __init__(self, modelPath: str, samplePath: str, displaySize=750, initialPredictionStep=30, modelSavePath="./ModifiedModel"):
        # 加载模型
        self.model = load_model(modelPath)
        # 加载样本数据
        sample = loadSample(samplePath)
        self.labels = sample[:, 2].astype('uint8')  # type: np.ndarray
        self.data = sample[:, :2].astype('float')  # type: np.ndarray
        self.exPoints = []
        self.exLabels = []  # 0 -> red, 1 -> green
        self.displaySize = displaySize
        self.predictionStep = initialPredictionStep
        self.allPt = None  # type: Optional[np.ndarray]
        self.allLb = None  # type: Optional[np.ndarray]
        self.fittingThread = threading.Thread(target=self.exFit, daemon=True)
        self.running = True
        self.modelSavePath = modelSavePath

        self._predictedPts = None  # type: Optional[np.ndarray]
        self._predictedGraph = None  # type: Optional[np.ndarray]
        self._loss = None
        self._acc = None
        self._predictionDrawingLock = threading.Lock()

    def exFit(self):
        # 增量训练模型
        while self.running:
            if self.allPt is None:
                continue

            history = self.model.fit(
                self.allPt / self.displaySize,
                self.allLb,
                epochs=100, batch_size=9999999
            )

            self._loss = history.history['loss'][-1]
            self._acc = history.history['accuracy'][-1]
            # 预测并绘制背景
            self._predictionDrawingLock.acquire()

            self._predictedPts, self._predictedGraph = getPredictionGraph(self.model, self.displaySize,
                                                                          self.predictionStep)
            self._predictionDrawingLock.release()
            pass

    def updateAllData(self):
        self.allPt = np.array(list(self.data * self.displaySize) + self.exPoints)
        self.allLb = np.array(list(self.labels) + self.exLabels)

    def main(self):
        # 创建画板
        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((self.displaySize, self.displaySize))
        screen.fill((255, 255, 255))

        pygame.font.init()
        font = pygame.font.SysFont("Consolas", 25)

        self.fittingThread.start()

        while self.running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.MOUSEWHEEL:
                    self.predictionStep -= event.y
                    self.predictionStep = self.bound(self.predictionStep, 1, 100)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == pygame.BUTTON_LEFT:
                        self.exPoints.append(event.pos)
                        self.exLabels.append(0)
                    elif event.button == pygame.BUTTON_RIGHT:
                        self.exPoints.append(event.pos)
                        self.exLabels.append(1)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LSHIFT:
                        if not self.exLabels:
                            continue
                        self.exLabels.pop(-1)
                        self.exPoints.pop(-1)
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        self.running = False
                        return
                    elif event.key == pygame.K_RETURN:
                        self.running = False
                        pygame.quit()
                        self.model.save(self.modelSavePath)
                        return
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    self.running = False
                    return
            self.updateAllData()  # 更新要显示的点
            # 绘制样本点
            if self._predictedPts is not None:
                self._predictionDrawingLock.acquire()
                for i, p in enumerate(self._predictedPts):
                    pygame.draw.rect(
                        screen,
                        (0, 0, self.bound(int(self.activateFunc(self._predictedGraph[i]) * 255), 0, 255)),
                        (p[0], p[1], self.predictionStep, self.predictionStep)
                    )
                    # ----
                    # if self._predictedGraph[i] < 0.5:
                    #     pygame.draw.rect(screen, (0, 0, 0), (p[0], p[1], self.predictionStep, self.predictionStep))
                    # else:
                    #     pygame.draw.rect(screen, (255, 255, 255),
                    #                      (p[0], p[1], self.predictionStep, self.predictionStep))
                self._predictionDrawingLock.release()
            # 绘制新增点
            if self.allLb is not None:
                for l, p in zip(self.allLb, self.allPt):
                    if l == 0:
                        pygame.draw.rect(screen, (255, 0, 0), (p[0], p[1], 5, 5))
                    else:
                        pygame.draw.rect(screen, (0, 255, 0), (p[0], p[1], 5, 5))
            # 输出fps
            if self._loss is not None:
                text = font.render(f"{self._loss:.2f} {self._acc:.2f}", 0, (0, 255, 0))
                textRect = text.get_rect()
                textRect.left = 10
                textRect.top = 10
                screen.blit(text, textRect)

            pygame.display.update()
            screen.fill((0, 0, 255))
            clock.tick(60)

    @classmethod
    def activateFunc(cls, val: float) -> float:
        """让临近0.5的值更加分散"""
        rst = math.tanh(
            (val - 0.5) * 10
        )
        rst += 1
        rst /= 2
        return rst

    @classmethod
    def bound(cls, val, lower, upper):
        """把某个值限制在区间内"""
        return max(lower, min(upper, val))


if __name__ == '__main__':
    Display("./trained.model", "./sample", 750, 200).main()
