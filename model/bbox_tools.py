import numpy as np


# 这个方法应该是给bbox和位置偏差，去调整bbox的位置
def loc2bbox(src_bbox, loc):
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    # copy设置为False，理解为在原src_bbox内存上修改
    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
    # src_bbox(ymin, xmin, ymax, xmax)
    # height是高，width是宽
    # 把(ymin, xmin, ymax, xmax)的左上右下转换成中心点坐标加宽和高
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    # 这里开始在想为什么不用 loc[:, 0]，后来试了一下发现用[:, 0::4]可以生成[R,1],
    #  而loc[:, 0]生成[1, R]
    # 这里也可以看出来loc里面放的是中心点坐标加宽和高的偏移量
    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    # np.newaxis是在原数组里加一个维度，我猜测这里的src_height等输出shape是一维数组(R,)
    # 在这里变为二维数组(R, 1)
    # 这里的计算公式在source图片里面有
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    # 接着把计算后的中心坐标和宽和高再转换成左上右下
    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    # 返回修正后的bbox
    return dst_bbox


# 这个方法是计算两个框之间的偏差
def bbox2loc(src_bbox, dst_bbox):
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    # 这里保证height，width不为0
    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


# 计算两个bbox之间的iou
# 我猜测这里的bbox_a是指在这张图上面所有的roi， bbox_b是指在这张图上的真实存在的ground truth
# 所有bbox_a有M个，bbox_b有N个，这个时候要用到广播，计算每个roi对应N个ground truth的iou
def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    # 假如bbox_a(5, 4), bbox_b(3, 4) label(3, 1)
    # 对bbox_a增加一维(5, 1, 4) maximun之后就变成了(5, 3, 2)
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    # area_i(5, 3) area_a(5, 1) area_b(3, 1)
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod((bbox_a[:, 2:] - bbox_a[:, :2]), axis=1)
    area_b = np.prod((bbox_b[:, 2:] - bbox_b[:, :2]), axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)
