import itertools
import numpy as np
import six

# defaultdict是当在字典里查找某个值没有找到时，返回一个默认值
from collections import defaultdict

from model.bbox_tools import bbox_iou


def eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
                       gt_difficults=None, iou_thresh=0.5, use_07_metric=False):
    # 计算prec和recall
    prec, rec = calc_detectioin_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)

    ap = calc_dectection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}

# pred_bboxes(1,200,4) pred_labels(1,200) pred_scores(1,200) gt_bboxes(1,5,4)
# gt_labels(1,5) gt_difficults(1,5)
def calc_detectioin_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults=None,
        iou_thresh=0.5):
    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    # defaultdict(int) 默认返回 0 defaultdict(list) 默认返回 []
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    # pred_bboxes(1,200,4) pred_labels(1,200) pred_scores(1,200) gt_bboxes(1,5,4)
    # gt_labels(1,5) gt_difficults(1,5)
    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in six.moves.zip(
            pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults):

        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            # True应该是留下，False删除
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]

            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            # 逻辑非
            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            # extend数组连接
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(axis=1)

            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        prec[l] = tp / (fp + tp)
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_dectection_voc_ap(prec, rec, use_07_metric=False):
    # n_fg_class = 20
    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            i = np.where(mrec[1:] != mrec[:-1])[0]

            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
