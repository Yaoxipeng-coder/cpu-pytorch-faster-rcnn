import torch
import torch.nn as nn
from torch.autograd import Function
import util.array_tool as at


class RoIPooling2D(nn.Module):
    def __init__(self, outh, outw, spatial_scale):
        super(RoIPooling2D, self).__init__()
        self.RoI = RoI(outh, outw, spatial_scale)

    def forward(self, x, rois):
        return self.RoI(x, rois)


class RoI(nn.Module):
    def __init__(self, outh, outw, spatial_scale):
        self.outh = outh
        self.outw = outw
        self.spatial_scale = spatial_scale

    def forward(self, features, rois):
        rois = at.totensor(rois).float()
        roi_list = []
        for roi in rois:
            # 这个计算的应该是按比例缩放roi尺寸到特征图上面
            # 比如原图为(300, 500) roi(34, 76, 82, 95),那么roi是在原图上框出来的框，现在要转换到特征图上的框
            # 特征图比如说是(37, 50) 给roi除以16的位置就是特征图所在的位置
            # roi / 16 = (34 / 16, 76 / 16, 82 / 16, 95 / 16) 求整 (2, 4, 5, 5)
            # 即对应到特征图的位置就是(2,4,5,5) 这里的位置就是坐标就是索引，即索引(2:5+1,4:5+1)是高和宽
            # 为什么加一，我猜是因为怕ymin和ymax值一样，长度为0了
            roi_part = features[:, :, (roi[0] / self.spatial_scale).int(): (roi[2] / self.spatial_scale).int() + 1,
                       (roi[3] / self.spatial_scale).int(): (roi[1] / self.spatial_scale).int() + 1]
            roi_part = nn.AdaptiveMaxPool2d((7, 1))(roi_part)
            roi_list.append(roi_part)
        pool = torch.cat(roi_list)  # B,C,7,7
        pool = pool.reshape(pool.shape[0], -1)  # B,C*7*7
        return pool
