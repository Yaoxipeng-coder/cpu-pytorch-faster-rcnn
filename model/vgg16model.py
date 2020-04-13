import torchvision
import torch
from torchvision import models
import torch.nn as nn
from model.faster_rcnn import FasterRCNN
from model.region_proposal_network import RegionProposalNetwork
from model.region_proposal_network import normal_init
from model.roi_module import RoIPooling2D
import util.array_tool as at


def decom_vgg16():
    vgg16 = models.vgg16(pretrained=True)
    # 卷积层删除最后一层，全连接层也删除最后一层, 固定前四层卷积层参数
    vgg16_features = list(vgg16.features)[:30]
    vgg16_classifier = list(vgg16.classifier)[:6]
    for layer in vgg16_features[:10]:
        for para in layer.parameters():
            para.requires_grad = False
    vgg16_features = nn.Sequential(*vgg16_features)
    vgg16_classifier = nn.Sequential(*vgg16_classifier)
    return vgg16_features, vgg16_classifier



# 在网络中出来的数都是预测的偏移量的差数，是要在预测的bbox上面加上这个偏移量的差数
class FasterRCNNVGG16(FasterRCNN):
    # feat_stride表示经过网络层后图片缩小为原图的 1/feat_stride
    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
        # 卷积层用来提取特征，全连接层用来分类
        extractor, classifier = decom_vgg16()

        # rpn = rpn_locs, rpn_scores, rois, anchor
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride, )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


# RoIhead主要任务是对RPN网络选出的候选框进行分类和回归
# 在特征图上
class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.cls_score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.cls_score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        # 7, 7, spatial_scale
        # self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois):
        rois = at.totensor(rois).float()
        roi_list = []
        for roi in rois:
            # 这个计算的应该是按比例缩放roi尺寸到特征图上面
            # 比如原图为(300, 500) roi(34, 76, 82, 95),那么roi是在原图上框出来的框，现在要转换到特征图上的框
            # 特征图比如说是(37, 50) 给roi除以16的位置就是特征图所在的位置
            # roi / 16 = (34 / 16, 76 / 16, 82 / 16, 95 / 16) 求整 (2, 4, 5, 5)
            # 即对应到特征图的位置就是(2,4,5,5) 这里的位置就是坐标就是索引，即索引(2:5+1,4:5+1)是高和宽
            # 为什么加一，我猜是因为怕ymin和ymax值一样，长度为0了
            roi_part = x[:, :, (roi[0] * self.spatial_scale).int(): (roi[2] * self.spatial_scale).int() + 1,
                       (roi[1] * self.spatial_scale).int(): (roi[3] * self.spatial_scale).int() + 1]
            roi_part = nn.AdaptiveMaxPool2d((7, 7))(roi_part)
            roi_list.append(roi_part)
        pool = torch.cat(roi_list)  # B,C,7,7
        pool = pool.reshape(pool.shape[0], -1)  # B,C*7*7

        # pool(B, C*7*7)
        # pool = self.roi(x, rois)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.cls_score(fc7)
        # 返回roi修正系数和分数
        return roi_cls_locs, roi_scores
