import numpy as np


# cupy在功能上与numpy基本无差，但是cupy是直接调用GPU运行计算，numpy是运行CPU运算

def non_maximun_suppression(bbox, thresh, score=None, limit=None):
    return _non_maximun_suppression_cpu(bbox, thresh, score, limit)


# 这个方法应该是实际执行nms的地方，用其他的边界框和预定的边界框比较，bbox(R, 4)
def _non_maximun_suppression_cpu(bbox, thresh, score, limit):  # 没有GPU，但是暂时用这个名字
    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    n_bbox = bbox.shape[0]

    if score is not None:
        # 分数由高到低排序
        order = score.argsort()[::-1].astype(np.int32)
    else:
        order = np.arange(n_bbox, dtype=np.int32)

    # 这里的sorted_bbox是按照分数从大到小排序
    # 应该删除这句话
    # sorted_bbox = bbox[order, :]
    # print(sorted_bbox.shape)
    # nms之后选择的bbox

    selectbbox = _call_nms_kernel_cpu(
        bbox, order, thresh
    )

    return selectbbox


def _call_nms_kernel_cpu(bbox, order, thresh):
    '''
    n_bbox = bbox.shape[0]
    threads_per_block = 64
    # np.ceil(a)取数组a中每个数大于本身的最小整数
    # col_blocks应该是取了多少块
    col_blocks = np.ceil(n_bbox / threads_per_block).astype(np.int32)
    # 括号代表元祖a = (1,2,3)，方括号代表数组b = [1,2,3]
    # 这里假设n_bbox为128，那么blocks(2,2,1) threads(64,1,1)
    blocks = (col_blocks, col_blocks, 1)
    threads = (threads_per_block, 1, 1)

    mask_dev = np.zeros((n_bbox * col_blocks,), dtype=np.int32)
    # np.ascontiguousarray是让数组中的元素按行变成顺序的 https://zhuanlan.zhihu.com/p/59767914
    bbox = np.ascontiguousarray(bbox, dtype=np.float32)
    kern = _load_kernel('nms_kernel', _nms_gpu_code)
    '''

    order = order
    y1 = bbox[:, 0]
    x1 = bbox[:, 1]
    y2 = bbox[:, 2]
    x2 = bbox[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)

        # 这里的xx1,yy1,xx2,yy2是为了求除第一个bbox以外所有bbox与第一个bbox的相交面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 这里加1有的人说是包含原坐标(0,0)点，暂时没看懂
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        # inter相交面积
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # np.where(ovr <= thresh)[0] 只保留iou不大于阈值的下标
        inds = np.where(ovr <= thresh)[0]
        # 这里因为inds是从第二个数开始的，比原order少第一个数，所以加1才能对上原order
        order = order[inds + 1]
    # keep在这里是一个下标索引值
    return keep
