import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from model.creater_tool import ProposalCreater


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32], feat_stride=16,
                 proposal_creater_params=dict(), ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        # feat_stride表示经过网络层后图片缩小为原图的1/feat_stride
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreater(self, **proposal_creater_params)
        # n_anchor = 9
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.score = nn.Conv2d(in_channels=mid_channels, out_channels=n_anchor * 2, kernel_size=1, stride=1, padding=0)
        self.loc = nn.Conv2d(in_channels=mid_channels, out_channels=n_anchor * 4, kernel_size=1, stride=1, padding=0)
        # 目前猜测这是对卷积层参数的归一化
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """
        Args:
        x(tensor): backbone输出的特征图.shape ->:math: `(N, C, H, W)`.
        img_size(tuple of ints): 元组: obj:`height, width`, 缩放后的图片尺寸.
        scale(float): 从文件读取的图片和输入的图片的比例大小.

        Returns:
        * ** rpn_locs **: 预测的anchor坐标位移.shape ->:math: `(N, H W A, 4)`.
        * ** rpn_scores **: 预测的前景概率得分.shape ->:math: `(N, H W A, 2)`.
        * ** rois **: 筛选后的anchor数组.它包含了一个批次的所有区域候选.shape ->:math: `(R', 4)`.
        * ** anchor **: 生成的所有anchor. \
        shape ->:math:`(H W A, 4)`.
        """
        # 这里的x应该是进行了卷积之后生成的特征图的大小，即为原图大小的1/16
        # x[B, C, H, W]
        n, _, hh, ww = x.shape
        # 生成在原图上的anchor,anchor(K * A, 4)
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
              self.feat_stride, hh, ww)
        # 这里的n_anchor猜测就是9， (hh * ww * 9) // (hh * ww)
        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))

        # 定位层，rpn_locs,这里的rpn_locs是预测的偏差，而不是预测位置
        rpn_locs = self.loc(h)  # (batch_size, 36, hh, ww)

        # 这句话是说对rpn_locs(batch_size, 36, hh, ww)先转换维度
        # 变为(batch_size, hh, ww, 36)再变为(batch_size, hh * ww * 9, 4)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        # rpn_scores(B, hh, ww, 18)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        # 这里应该是对第五维的数据进行softmax，应该是背景概率和前景概率的概率之和为1
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        # fg即foreground前景,前景估计是指有目标的位置，区别于背景，取得有前景的概率
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        # rpn_fg_scores(n, hh * ww * n_anchor * 1)
        #         # rpn_scores(n, hh * ww * n_anchor, 2)
        #         # rpn_scores可以看出来是每张anchor的前景背景概率
        #         # rpn_fg_scores是前景概率，每一批次里面放了hh * ww * n_anchor * 1个概率值
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        # rois是生成的anchor中选择有用的anchor
        rois = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].data.numpy(),
                rpn_fg_scores[i].data.numpy(),
                anchor, img_size,
                scale=scale,
            )
            # 把每次循环筛选出的bbox添加到rois中
            rois.append(roi)

        rois = np.concatenate(rois, axis=0)
        # hh, ww为37 * 50时
        # rpn_locs torch.Size([1, 16650, 4]) rpn_scores torch.Size([1, 16650, 2]) rois(1901, 4),
        # anchor(16650, 4)

        return rpn_locs, rpn_scores, rois, anchor


# 这个是产生所有anchor的地方,具体不懂的可以看 https://blog.csdn.net/qq_22526061/article/details/88574822
# 这个方法应该是把特征图的anchor转换到原图的anchor
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # shift_y.shape:[height,] shift_x.shape:[width,]
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)

    # np.meshgrid根据x,y生成网格点矩阵坐标
    # https://blog.csdn.net/LoveL_T/article/details/84328683?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
    # 简单来说，就是在x轴生成shift_x的点，在y轴上生成shift_y的点，在这个二维空间中(x,y)即为坐标点
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # np.stack()连接多个数组 ravel()转换为一维数组
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    # A = 4, K = height * width
    A = anchor_base.shape[0]
    K = shift.shape[0]
    # shift.shape:[height * width, 1, 4]
    anchor = anchor_base.reshape(1, A, 4) + shift.reshape(1, K, 4).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


# 产生anchor
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            # h代表高，w代表宽
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j  # 0,1,2,3,4,5,6,7,8 生成9个框
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    # 返回值是(ymin, xmin, ymax, xmax) 注意y在前
    return anchor_base


'''
a = generate_anchor_base()
print(a)
结果：
[[ -37.25483322  -82.50966644   53.25483322   98.50966644]
 [ -82.50966644 -173.01933289   98.50966644  189.01933289]
 [-173.01933289 -354.03866577  189.01933289  370.03866577]
 [ -56.          -56.           72.           72.        ]
 [-120.         -120.          136.          136.        ]
 [-248.         -248.          264.          264.        ]
 [ -82.50966644  -37.25483322   98.50966644   53.25483322]
 [-173.01933289  -82.50966644  189.01933289   98.50966644]
 [-354.03866577 -173.01933289  370.03866577  189.01933289]]
'''


# 权重的随机初始化，符合均值为mean,方差为stddev
def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
