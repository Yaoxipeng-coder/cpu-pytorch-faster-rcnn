import time
import os
import numpy as np
import torch.nn as nn
import torch
from data.config import opt
from model.creater_tool import AnchorTargetCreator
from model.creater_tool import ProposalTargetCreator
import torch.nn.functional as F
from util import array_tool as at


class FasterRCNNTrainer(nn.Module):
    # 封装一个训练类，返回损失值
    # 共五个损失值
    # rpn_loc_loss rpn_cls_loss roi_loc_loss roi_cls_loss total_loss
    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        # 3.
        self.rpn_sigma = opt.rpn_sigma
        # 1.
        self.roi_sigma = opt.roi_sigma

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()

    def forward(self, imgs, bboxes, labels, scale):
        # 每次进来一张图,对应的bbox，对应的label
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)
        # 求出特征图
        features = self.faster_rcnn.extractor(imgs)
        # 求出rpn网络输出的损失，roi，所有anchor
        # rpn_locs(N, H W A, 4)
        rpn_locs, rpn_scores, rois, anchor = self.faster_rcnn.rpn(features, img_size, scale)
        # bbox(R, 4) label(R, 1)
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois
        # bbox和label是真实的bbox和label
        # roi是预测的bbox
        # 返回的是选择后的roi以及roi对应真实bbox的位置偏差，以及roi所给的label
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)

        # 应该是返回在特征图上的roi全连接之后的位置和概率
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi)

        # rpn loss
        # 给anchor标记标签，loc上默认值是0，正样本为位移偏移量
        # label上默认值为-1，正样本为1
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)

        # rpn_loc(H * W * A, 4)
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma)
        # 计算rpn网络的分类损失，忽略label = -1的
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

        # ROI loss
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long(), at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc,
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

        losses = {'rpn_loc_loss': rpn_loc_loss,
                  'rpn_cls_loss': rpn_cls_loss,
                  'roi_loc_loss': roi_loc_loss,
                  'roi_cls_loss': roi_cls_loss}
        losses['total_loss'] = sum(losses.values())
        return losses

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses['total_loss'].backward()
        self.optimizer.step()
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, save_path)
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    # 猜测形状应该是pred_loc(H * W * A, 4), gt_loc(H * W * A, 4), gt_label(H * W * A, 1)
    in_weight = torch.zeros(gt_loc.shape)
    # expand_as 把tensor的形状变为和括号内的tensor一样的形状
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss
