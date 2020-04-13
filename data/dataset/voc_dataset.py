from lxml import etree
import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
# filename size object
# 网络需要两个东西，一个是图片的信息包括 C, H, W 一个是图片的灰度值
from data.util import read_imagebycv2


class VOCBboxDataset:
    def __init__(self, filepath, split='trainval', use_difficult=False,
                 return_difficult=False):
        self.filepath = filepath
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        id_list_file = os.path.join(filepath, r'ImageSets\Main\{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]

    def __len__(self):
        return len(self.ids)

    def get_example(self, filenameindex):
        id_ = self.ids[filenameindex]
        bbox = list()
        label = list()
        difficult = list()
        anno = os.path.join(self.filepath, r'Annotations\\' + str(id_) + '.xml')
        anno = ET.parse(anno)
        for obj in anno.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.float32)

        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)
        img_path = r'C:\Users\Administrator\PycharmProjects\faster-rcnn-practice\data\VOCdevkit\VOC2007\JPEGImages'
        img_file = os.path.join(img_path, str(id_) + '.jpg')
        # H, W, 3
        img = cv2.imread(img_file)
        return img, bbox, label, difficult

    # __getitem__是特殊方法，是用来迭代索引的
    __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
