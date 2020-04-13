import torch
import cv2
import numpy as np


# 随机产生人工椒盐
def salt(img):
    for k in range(1500):
        # W
        i = int(np.random.random() * img.shape[1])
        # H
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255
    return img


# img = cv2.imread(r'C:\Users\Administrator\PycharmProjects\faster-rcnn-practice\data\dataset\bb36cbd09d89e105ff5c4096d0c1f6e.jpg')

'''
hist = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])
cv2.imshow('hist', hist)
cv2.waitKey()
H, W = img.shape[:2]
img = cv2.resize(img, (int(W / 2), int(H / 2)))
img = salt(img)
cv2.imshow('img', img)
cv2.waitKey()


'''
