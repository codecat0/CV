# Faster-RCNN
## 1. 项目描述
1. 通过Pytorch简单实现了[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
2. 参考
    - [https://github.com/chenyuntc/simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)
    - [https://github.com/bubbliiiing/faster-rcnn-pytorch](https://github.com/bubbliiiing/faster-rcnn-pytorch)

## 2. 代码结构
```
|--dataset
|    |--faster_rcnn_dataset.py : faster-rcnn数据集操作
|    |--utils.py : 数据集相关配置
|--backbones
|    |--resnet.py : ResNet网络提取特征
|    |--vggnet.py : VGG网络提取特征
|--rpn   
|    |--proposal_create.py : RPN网络中Proposal层，用来生成proposals
|    |--rpn_net.py : RPN网络，进行先验框的初步筛选
|--roi_head
|    |--resnet_roi_head.py : ResNet对应的RoIHead，用来计算筛选后proposals的分类参数和回归参数
|    |--vgg_roi_head.py : VGG对应的RoIHead，用来计算筛选后proposals的分类参数和回归参数
|--nets
|    |--faster_rcnn.py : Faster RCNN网络
|    |--frcnn_training.py : Faster RCNN训练器，包含了给先验框anchor匹配对应的gtbox以及各候选框proposals匹配对应的gtbox
|--utils
|    |--anchor_utils.py : anchor操作相关配置
|    |--box_utils.py : box框操作相关配置
|    |--data_utils.py ： 数据处理相关配置
|    |--metric_utils.py : 评价指标相关配置
|    |--train_utils.py : 训练相关配置
|    |--callbacks.py : 记录日志相关操作
|--model_data
|    |--train.txt ： 训练时使用，每行包含训练图片路径，每张图片中gtbox的位置以及对应的类别
|    |--val.txt ： 验证时使用，每行包含验证图片路径，每张图片中gtbox的位置以及对应的类别
|    |--voc_classes.txt : 类别文件，每行对应一个类别
|    |--simhei.ttf : 字体文件
|--voc_annotations.py : 用来生成model_data下的train.txt和val.txt
|--train.py : 训练脚本
|--predict.py : 预测脚本
|--get_map : 获取评价指标
```

## 3. 数据集
   - **VOC数据下载地址：** [http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/)

## 4. 环境配置
```
numpy==1.21.2
torch==1.9.1
torchvision==0.11.1
pillow==8.3.1
opencv-python==4.5.4.58
scipy==1.7.2
matplotlib==3.4.3
tqdm==4.62.3
```

## 5. 模型效果展示
![](img/1_dr.jpg)

## 6. 训练自己的模型
1. 数据集的准备
   - 训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。路径对应于`.gitignore`文件的第2行
   - 训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。路径对应于`.gitignore`文件的第2行

2. 数据集的处理
   在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的train.txt和val.txt。**注意修改classes_path**

3. 开始网络训练
   **注意修改classes_path**，运行`train.py`即可
4. 训练结果预测
   - `model_path`指向训练好的权值文件
   - `classes_path`指向检测类别所对应的txt
   - 修改后就可以运行`predict.py`进行检测了

