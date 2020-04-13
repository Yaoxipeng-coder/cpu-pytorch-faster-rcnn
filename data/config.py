from pprint import pprint


class Config:
    # data
    voc_data_dir = r'C:\Users\Administrator\PycharmProjects\faster-rcnn-practice\data\VOCdevkit\VOC2007'
    test_image_dir = r'C:\Users\Administrator\PycharmProjects\faster-rcnn-practice\trainandtest\modeltestimage'
    save_image_dir = r'C:\Users\Administrator\PycharmProjects\faster-rcnn-practice\trainandtest\modelsaveimage\\'
    min_size = 600
    max_size = 1000
    num_workers = 0
    test_num_workers = 0

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster_rcnn
    # 训练时候选择
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e_3 -> 1e-4
    lr = 1e-3

    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 40

    data = 'voc'
    pretrained_model = 'vgg16'

    epoch = 2

    use_adam = False
    use_chainer = False
    use_drop = False

    test_num = 1000

    load_path = None

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('=====user config======')
        print(self._state_dict())
        print('========end=======')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() if not k.startswith('_')}


opt = Config()
