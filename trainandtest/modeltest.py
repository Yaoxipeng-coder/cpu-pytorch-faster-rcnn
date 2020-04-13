import torch
import numpy as np
import cv2
from data.dataset.dataset import preprocess
from data.config import opt
import os
from torch.utils import data
from model.vgg16model import FasterRCNNVGG16
from data.dataset.voc_dataset import VOC_BBOX_LABEL_NAMES


class Dataset:
    def __init__(self, filepath):
        self.filepath = filepath

    def __len__(self):
        return len(os.listdir(self.filepath))

    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.filepath, str(item + 1) + '.jpg'))
        init_img = img
        process_img = preprocess(img)
        return process_img, init_img


def train():
    path = r'C:\Users\Administrator\PycharmProjects\faster-rcnn-practice\trainandtest\checkpoints\fasterrcnn_041108170.159090909091'
    names = VOC_BBOX_LABEL_NAMES
    # 模型框架
    model = FasterRCNNVGG16()
    # 模型参数
    checpoint = torch.load(path)
    model.load_state_dict(checpoint['model'])

    '''
    print(model)
    for i in model.named_parameters():
        print(i)
    '''

    testImage = Dataset(opt.test_image_dir)
    testdata = data.DataLoader(testImage, batch_size=1)
    for ii, (img, init_img) in enumerate(testdata):
        H = img.shape[2]
        W = img.shape[3]
        scale = init_img.shape[1] / H
        size = [H, W]
        bboxes, labels, scores = model.predict(imgs=img, sizes=[size], visualize=True)
        bboxes = np.array(bboxes[0])
        labels = np.array(labels[0])
        scores = np.array(scores[0])
        imgplay = init_img[0].numpy()
        for id in range(bboxes.shape[0]):
            bbox_copy = (bboxes[id] * scale).copy()
            label_copy = labels[id].copy()
            score_copy = scores[id].copy()
            img_copy = imgplay.copy()
            imgplay = cv2.rectangle(img_copy, (bbox_copy[1], bbox_copy[0]), (bbox_copy[3], bbox_copy[2]), (0, 0, 255),
                                    2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            label_copy = names[label_copy]
            text = str(label_copy) + ' ' + str(score_copy)
            imgplay = cv2.putText(imgplay, text, (bbox_copy[1], int(bbox_copy[0] + 20)), font, 1, (255, 0, 0), 2,
                                  cv2.LINE_AA)
        cv2.imwrite(opt.save_image_dir + str((ii + 1)) + '.jpg', imgplay)
        # cv2.imshow('img', imgplay)
        # cv2.waitKey()


if __name__ == '__main__':
    train()
