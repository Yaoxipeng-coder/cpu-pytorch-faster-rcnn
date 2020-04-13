from data.dataset.voc_dataset import VOCBboxDataset
import numpy as np
import torch
from data.config import opt
import torchvision
from torchvision import transforms
from data import util
import torch.nn.functional as F
import cv2


def preprocess(img, min_size=600, max_size=1000):
    H, W, C = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    totensor = transforms.ToTensor()
    img = totensor(img)
    #resize = transforms.Resize((H * scale, W * scale), interpolation='Image.NEAREST')
    img = F.interpolate(img.unsqueeze(0), size=(round(H * scale), round(W * scale)), mode="nearest").squeeze(0)
    #img = resize(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(img)
    img = img.numpy()
    # numpy 使用 transpose交换列
    # pytorch使用permute交换列
    #img = img.transpose(1, 2, 0)
    #img = img.numpy()
    #cv2.imshow('image', img)
    #cv2.waitKey()
    #img = img.transpose(2, 0, 1)
    return img


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        H, W, _ = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))
        # 旋转图片
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])
        return img, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        img = np.ascontiguousarray(img)
        bbox = np.ascontiguousarray(bbox)
        label = np.ascontiguousarray(label)
        img = torch.from_numpy(img)
        # img, bbox, label = torch.from_numpy(img), torch.from_numpy(bbox), torch.from_numpy(label)
        return img.clone(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='val', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[:2], bbox, label, difficult

    def __len__(self):
        return len(self.db)