import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from .nms import non_maximun_suppression
from data.config import opt
from data.dataset.dataset import preprocess
from util import array_tool as at
from model.bbox_tools import loc2bbox


def nograd(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)

    return new_f


# 基础FasterRCNN模型
class FasterRCNN(nn.Module):
    # 3个参数分别是特征提取，region proposal network， roi
    # loc_normalize_mean, loc_normalize_std保留疑问
    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)
                 ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        # 使用训练状态下的阈值
        self.use_preset('evaluate')

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    # scale=1.应该是缩放比
    def forward(self, x, scale=1.):

        # 获取图片的H,W
        img_size = x.shape[2:]

        # 特征提取
        h = self.extractor(x)

        # rpn rpn_locs是预测的anchor坐标位移(B, H, W, A, 4)，rpn_scores是预测的anchor概率得分(B, H, W, A, 2) A是每个像素产生的anchor个数
        rpn_locs, rpn_scores, rois, anchor = self.rpn(h, img_size, scale)
        # roi
        roi_cls_locs, roi_scores = self.head(h, rois)
        # return里面没有rpn_locs和rpn_scores，那这两个有什么用？
        return roi_cls_locs, roi_scores, rois

    # 这个函数是两种状态，evaluate是在训练状态下的阈值,visualize是在显示图片效果时的阈值
    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.2
            self.score_thresh = 0.5
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    # 这个应该是nms中的s，即抑制
    # raw_cls_bbox (84,4)
    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # 跳过cls_id = 0 因为 等于0时类别是背景
        for l in range(1, self.n_class):
            # 在这里猜测后面的[:, l, 4]意思是取每个类别的bbox
            # 比如假设raw_cls_bbox.reshape之后形状是[20,21,4],取[:, l, 4]之后是[20,1,4]
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            # 这个是每个类别每个框的概率 比如[100, 21]变为[100, 1]
            prob_l = raw_prob[:, l]
            # 分数大于阈值的框被留下来，小的删掉 cls_bbox_l[mask]当mask为true不变，当mask为false删掉
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            # 用到了另外的模块nms和cupy,这里暂时没有cupy，没有GPU
            keep = non_maximun_suppression(
                np.array(cls_bbox_l), self.nms_thresh, prob_l  # 原本是import cupy as cp
            )

            keep = np.array(keep)  # 同上
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2]. 这里应该是取掉了背景这一分类
            # 这个应该是给bbox标记所属类别，每一次的for循环里面的标签是一样的
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        # np.concatenate()是数组的拼接，这里返回值应该是bbox(R, 4), label(R,), score(R,)
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    def get_optimizer(self):
        lr = opt.lr
        params = []
        # 获得训练参数
        # 获取梯度更新的方式,以及 放大 对网络权重中 偏置项 的学习率
        for key, value in dict(self.named_parameters()).items():

            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    # 这个是在测试阶段需要调的函数，用来计算训练完成模型之后得预测情况
    @nograd
    def predict(self, imgs, sizes=None, visualize=False):
        """
        在计算mAP的时候使用
        :param imgs: 一个batch的图片
        :param sizes: batch中每张图片的输入尺寸
        :return: 返回所有一个batch中所有图片的坐标,类,类概率值 三个值都是list型数据,里面包含的是numpy数据
        """

        self.eval()
        if visualize:
            self.use_preset('visualize')
        prepared_imgs = imgs

        bboxes = list()
        labels = list()
        scores = list()

        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois = self(img, scale=scale)
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale

            mean = torch.Tensor(self.loc_normalize_mean).repeat(self.n_class)[None]
            std = torch.Tensor(self.loc_normalize_std).repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            # 从这里其实可以看出，roi_cls_loc是位置偏差,是用来计算更接近的位置,而不是位置,
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # 这里应该是控制cls_bbox的边界,在min和max之间
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))

            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)
            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        # 这里返回的是预测图片的bbox，labels，scores
        return bboxes, labels, scores

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
