import numpy as np

from model.nms import non_maximun_suppression
# 这个类在做什么还没看懂
from model.bbox_tools import loc2bbox, bbox_iou, bbox2loc


class AnchorTargetCreator(object):

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        img_H, img_W = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(inside_index,
                                                anchor, bbox)
        # loc(H * W * A, 4)
        loc = bbox2loc(anchor, bbox[argmax_ious])
        label = _unmap(label, n_anchor, inside_index)
        loc = _unmap(loc, n_anchor, inside_index)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label为1是正样本，0负样本，-1不管
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)

        label[max_ious < self.neg_iou_thresh] = 0

        label[gt_argmax_ious] = 1

        label[max_ious >= self.pos_iou_thresh] = 1

        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos),
                                             replace=False)
            label[disable_index] = -1

        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg),
                                             replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]

        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    if len(data.shape) == 1:
        # 如果是对label进行映射，则默认值为-1(忽略样本)
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        # 如果是对loc进行映射，则默认值为0(忽略样本)
        # ret(N, 4)
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    # 计算anchor框的坐标都是位于图片内部的
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


class ProposalTargetCreator(object):
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_noramlize_std=(0.1, 0.1, 0.2, 0.2)):
        n_bbox, _ = bbox.shape
        roi = np.concatenate((roi, bbox), axis=0)
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        # iou(N, K) N > K
        iou = bbox_iou(roi, bbox)
        # 求出sample bbox与哪个ground truth之间的iou最大
        # gt_assignment是位置的索引,max_iou是真实iou值 gt_assignment(N,)
        # gt_assignment的索引值范围是(0, K-1) 它刚好是label的取值范围
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)

        # 这个就是给sample bbox类别标记一个分类类型,加一是给0留位置，0后面会表示背景
        # gt_roi_label就是给每个roi标记了label标签，label是它与哪个label的iou值最大的标签
        gt_roi_label = label[gt_assignment] + 1
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
        # 正样本标记为各个分类的类型，负样本标记为0
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        sample_roi = roi[keep_index]

        # 计算修正系数    roi和其最大iou的target_box的loc
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_noramlize_std, np.float32))
        return sample_roi, gt_roi_loc, gt_roi_label


class ProposalCreater(object):
    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, image_size, scale=1.):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        # 返回修正后的bbox(R, 4)
        roi = loc2bbox(anchor, loc)

        '''
        生成图片太多，通过一些手段删除些框
        图像外面的框进行裁剪
        去除宽或高小于给定阈值的框
        对这些roi根据score进行降序排序，取top 12000(测试时top 6000)
        nms之后取出top 2000(测试时top300)
        到这里，外面得到了2000个(测试时是300)个候选框
        '''
        # slice(start, stop, step) np.clip(array, 最小值, 最大值)
        # 这个应该是在筛选bbox的坐标应该在图片的范围内，坐标超过范围，太大
        # 或太小都删去
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, image_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, image_size[1])

        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # 这里的order得到的也是一个下标索引值
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        keep = non_maximun_suppression(
            np.ascontiguousarray(np.asarray(roi)),
            thresh=self.nms_thresh
        )
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep]
        # 至此bbox从生成多个到筛选出来的bbox才完成
        return roi
