### project 借鉴了https://github.com/chenyuntc/simple-faster-rcnn-pytorch
### cpu-pytorch-faster-rcnn
### 下载voc2007数据集放在 faster-rcnn-practice\data\dataset目录下
### 在\trainandtest\train.py下训练数据，在\trainandtest\modeltest.py下测试数据
### 实现了faster-rcnn的cpu版，原代码中涉及到的c语言，由于pytorch语言不再兼容c语言版的，改为python语言。
### 由于本人暂时没有GPU条件，cpu修改后实验跑了2000张训练集，效果还可以，准备后面再跑数据，后面再学习GPU版的操作，后续会放入一个更多训练之后的网络
### 里面有一张实例：原图为cpu-pytorch-faster-rcnn/trainandtest/modeltestimage/1.jpg
### 检测图片效果展示为cpu-pytorch-faster-rcnn/trainandtest/modelsaveimage/1.jpg
####在2000张训练集之后的表现，可以训练更多数据会有更好的效果
![image](https://github.com/Yaoxipeng-coder/cpu-pytorch-faster-rcnn/blob/master/trainandtest/modelsaveimage/1.jpg)
