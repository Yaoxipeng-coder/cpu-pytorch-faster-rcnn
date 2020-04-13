from data.config import opt
from data.dataset.dataset import Dataset, TestDataset
from torch.utils import data as data_
from model.vgg16model import FasterRCNNVGG16
from trainandtest.trainer import FasterRCNNTrainer
from tqdm import tqdm
import numpy as np
import torch
from util import array_tool as at
from util.eval_tool import eval_detection_voc
import cv2


def eval(dataloader, faster_rcnn, test_num=1000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    # len(dataloader) = 2510
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        # item是用来访问Tensor里面的元素
        # x = torch.randn(1)
        # print(x)
        # print(x.item())
        #
        # 结果是
        # tensor([-0.4464])
        # -0.44643348455429077
        '''
        if ii == 1:
            break
        '''
        # cv2.imshow('img', imgs[0].permute(1, 2, 0).numpy())
        # cv2.waitKey()
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        # imgs_copy = imgs[0].permute(1, 2, 0).numpy().copy()
        # img_withrec = cv2.rectangle(imgs_copy, (np.array(pred_bboxes_)[0][0][0], np.array(pred_bboxes_)[0][0][1]), (np.array(pred_bboxes_)[0][0][2], np.array(pred_bboxes_)[0][0][3]), (0, 0, 255), 1)
        # cv2.imshow('imgs', img_withrec)
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num:
            break
    # pred_bboxes(1,200,4) pred_labels(1,200) pred_scores(1,200) gt_bboxes(1,5,4)
    # gt_labels(1,5) gt_difficults(1,5)
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


if __name__ == '__main__':
    # opt._parse()

    # len(dataset) = 2501
    dataset = Dataset(opt)
    print('load data')
    # 关于数据集，这里的数据是指定路径下的所有图片(H, W),
    # bbox(ymin, xmin, ymax, xmax) label是bbox对应的类别
    # torch.util.data.DataLoader作为pytorch的数据加载器，所需要的
    # dataset数据类型是tensor的
    traindataloader = data_.DataLoader(dataset,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=opt.num_workers)
    # len(testset) = 2510
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    faster_rcnn = FasterRCNNVGG16()
    print('model contruct completed')
    trainer = FasterRCNNTrainer(faster_rcnn)
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    best_map = 0
    lr_ = opt.lr
    # opt.epoch = 14
    for epoch in range(opt.epoch):
        for ii, (img, bbox, label, scale) in tqdm(enumerate(traindataloader)):
            # img.shape = (1, C, H, W)
            '''
            if ii == 2000:
                break
            '''
            scale = at.scalar(scale)
            #img, bbox, label = img.numpy(), bbox.numpy(), label.numpy()
            trainer.train_step(img, bbox, label, scale)
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        print('**', eval_result)
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']

        if eval_result['map'] >= best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 3:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr = lr_ * opt.lr_decay

        if epoch == 4:
            break
