# encoding=utf-8
import pickle

import numpy as np
import pygame
from typing import List, Tuple, Sequence

import sklearn.datasets

size = 750


def editPointSample():
    pygame.init()
    redPoints = []
    greenPoints = []
    traceBackPoints = []  # 0 -> red, 1 -> green

    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((size, size))
    screen.fill((255, 255, 255))
    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == pygame.BUTTON_LEFT:
                    redPoints.append(event.pos)
                    traceBackPoints.append(0)
                    pygame.draw.rect(screen, (255, 0, 0), (event.pos[0], event.pos[1], 5, 5))
                elif event.button == pygame.BUTTON_RIGHT:
                    greenPoints.append(event.pos)
                    traceBackPoints.append(1)
                    pygame.draw.rect(screen, (0, 255, 0), (event.pos[0], event.pos[1], 5, 5))
                percentage = len(redPoints) / (len(redPoints) + len(greenPoints))
                print(f"{percentage:.2%}")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LSHIFT:
                    lastPointRG = traceBackPoints.pop(-1)
                    if lastPointRG == 0:
                        if redPoints:
                            pos = redPoints.pop(-1)
                            pygame.draw.rect(screen, (255, 255, 255), (pos[0], pos[1], 5, 5))
                    elif lastPointRG == 1:
                        if greenPoints:
                            pos = greenPoints.pop(-1)
                            pygame.draw.rect(screen, (255, 255, 255), (pos[0], pos[1], 5, 5))
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return redPoints, greenPoints
            elif event.type == pygame.QUIT:
                pygame.quit()
                return redPoints, greenPoints
        pygame.display.update()
        clock.tick(60)


def save(redPoints: List[Tuple[int, int]], greenPoints: List[Tuple[int, int]]):
    arr = np.zeros((len(redPoints) + len(greenPoints), 3))
    for index, point in enumerate(redPoints):
        arr[index, 0] = point[0]
        arr[index, 1] = point[1]
        arr[index, 2] = 0
    for index, point in enumerate(greenPoints):
        index += len(redPoints)
        arr[index, 0] = point[0]
        arr[index, 1] = point[1]
        arr[index, 2] = size
    arr = arr.astype('float') / size
    with open("sample", 'wb') as w:
        pickle.dump(arr, w)


def main():
    save(*editPointSample())


if __name__ == '__main__':
    main()
