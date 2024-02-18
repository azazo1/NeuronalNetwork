import sys
import threading
import time
from typing import Optional

import numpy as np
import pygame.display
from PIL import Image
from PIL.Image import Resampling
from pygame import Rect, Surface

sys.path.append("../..")
from imgcla.cifar.cnn_train import SimpleCNNModel


class Locker:
    def __init__(self):
        self.lock = threading.Lock()

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()


class Grid:
    def __init__(self, size: int):
        self.grid = np.zeros((size, size), dtype="uint8")
        self.size = size

    def bound(self, _min, _max, val):
        return min(_max, max(_min, val))

    def clear(self):
        self.grid.fill(0)

    def set(self, pos, value):
        self.grid[int(self.bound(0, self.size - 1, pos[1])), int(self.bound(0, self.size - 1, pos[0]))] = value

    def setRelatively(self, screenSize, rPos, value):
        """利用相对大小计算应设置的格子"""
        pixelPerGrid = screenSize / self.size
        x = rPos[0] // pixelPerGrid
        y = rPos[1] // pixelPerGrid
        self.set((x, y), value)

    def get(self, pos):
        return self.grid[int(self.bound(0, self.size, pos[1])), int(self.bound(0, self.size, pos[0]))]

    def getRelatively(self, screenSize, rPos):
        """利用相对大小计算获取的格子"""
        pixelPerGrid = screenSize / self.size
        x = rPos[0] // pixelPerGrid
        y = rPos[1] // pixelPerGrid
        return self.get((x, y))


class Predictor:
    def __init__(self, size=256):
        self.model = SimpleCNNModel()
        self.model.load("VGGTrainedMnistModel")
        self.screenSize = size
        self.ratio = self.screenSize // 28
        self.grid = Grid(28)
        self.surface = Surface((self.screenSize, self.screenSize))
        self.screen = None  # type: Optional[Surface]
        self.running = True
        self.predictingThread = threading.Thread(target=self.predict, daemon=True)
        self.predictionRst = "Result will be here"
        self.surfaceLock = Locker()

        self.predictingThread.start()

    def predict(self):
        while self.running:
            img = Image.new("1", (28, 28), (0,))
            for y in range(28):
                for x in range(28):
                    img.putpixel((x, y), int(self.grid.get((x, y))))
            img = img.resize((32, 32), resample=Resampling.NEAREST)
            batch = np.array(img).reshape((32, 32, 1))  # type: np.ndarray
            batch = batch.astype('float16') / 255

            prediction = self.model.predict([batch])
            print(prediction)
            num = prediction.argmax()
            percentage = prediction[0][num]
            self.predictionRst = f"{num}, {percentage:.2%}"

            time.sleep(0.5)

    def main(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screenSize, self.screenSize))
        while self.running:
            pygame.display.set_caption(self.predictionRst)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.running = False
                    return
                elif event.type == pygame.MOUSEMOTION:
                    pressed, *_ = pygame.mouse.get_pressed()
                    if pressed:
                        self.grid.setRelatively(self.screenSize, event.pos, 255)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.grid.clear()
            with self.surfaceLock:
                for x in range(self.grid.size):
                    for y in range(self.grid.size):
                        val = self.grid.get((x, y))
                        pygame.draw.rect(self.surface, (val, val, val),
                                         (x * self.ratio, y * self.ratio, self.ratio, self.ratio))
            with self.surfaceLock:
                self.screen.blit(self.surface, self.surface.get_rect())
            pygame.display.update()
            time.sleep(0.016)


if __name__ == '__main__':
    Predictor().main()
