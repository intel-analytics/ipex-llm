# MMCV Faster-RCNN Training Example

## Prepare the dataset

This training example uses the kitti_tiny dataset, you can download it from [here](https://download.openmmlab.com/mmdetection/data/kitti_tiny.zip) and unzip the dataset.

## Prepare the environment

We recommend you to use Anaconda to prepare the environment:

```python
conda create -n bigdl-orca python=3.7  # "bigdl-orca" is conda environment name, you can use any name you like.
conda activate bigdl-orca

pip install torch torchvision tensorboard
pip install -U openmim
mim install mmcv-full
pip install mmdet

pip install --pre --upgrade bigdl-orca[ray]
```

## Download MMDet

In this example, we need to use the config file from the [mmdet repo](https://github.com/open-mmlab/mmdetection).

```
git clone https://github.com/open-mmlab/mmdetection
```

The config file we use in this example is `mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py`

If you want to use the pre-trained checkpoints for further training, also download the checkpoint file:

```
cd mmdetection
mkdir checkpoints
wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth
```

## Running commands

- spark local mode

```bash
python train.py  --dataset /your/dataset/path/to/kitti_tiny/ --config $MMDET_PATH/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py --load_from $MMDET_PATH/checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth
```
`--dataset`: your local dataset path to kitti_tiny

`--config`: the local config file path, here we use the config file from mmdet repo

`--load_from`: the local pre-trained model path

`$MMDET_PATH`: your mmdetection repo local path.


- yarn-client mode

To train with yarn-client mode, you need to make sure the dataset is under the same path of every worker node. In order to do this, you can upload the dataset to HDFS and download dataset to every worker node before training.

To prepare dataset under every cluster node, you need to prepare the compressed package of the dataset and put it under the corresponding archive.

If you want to use the pre-trained checkpoints, please first upload the checkpoint file to HDFS, the checkpoint file can be directly read from HDFS.

Finally execute:

```bash
python train.py  --dataset /your/dataset/path/to/kitti_tiny/ --cluster_mode yarn-client --config $MMDET_PATH/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py --load_from hdfs://ip:port/your/hdfs/path/to/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth
```

`--cluster_mode`: The cluster mode, such as local, yarn-client, yarn-cluster, here we use yarn-client

`--dataset`: worker node local dataset path to kitti_tiny

`--config`: config file local path on driver node, here we use the config file from mmdet repo

`--load_from`: the pre-trained model path on HDFS

`$MMDET_PATH`: your mmdetection repo local path on driver node.


## Results

```
(MMCVRayEpochRunner pid=188889)
(MMCVRayEpochRunner pid=188889) ---------------iou_thr: 0.5---------------
(MMCVRayEpochRunner pid=188889) 2023-03-13 10:48:45,974 - mmdet - INFO -
(MMCVRayEpochRunner pid=188889) +------------+-----+------+--------+-------+
(MMCVRayEpochRunner pid=188889) | class      | gts | dets | recall | ap    |
(MMCVRayEpochRunner pid=188889) +------------+-----+------+--------+-------+
(MMCVRayEpochRunner pid=188889) | Car        | 62  | 632  | 0.984  | 0.791 |
(MMCVRayEpochRunner pid=188889) | Pedestrian | 13  | 78   | 0.846  | 0.800 |
(MMCVRayEpochRunner pid=188889) | Cyclist    | 7   | 66   | 0.286  | 0.052 |
(MMCVRayEpochRunner pid=188889) +------------+-----+------+--------+-------+
(MMCVRayEpochRunner pid=188889) | mAP        |     |      |        | 0.548 |
(MMCVRayEpochRunner pid=188889) +------------+-----+------+--------+-------+
(MMCVRayEpochRunner pid=188889) 2023-03-13 10:48:45,976 - mmdet - INFO - Epoch [1][25/25]       lr: 2.500e-03, eta: -1 day, 23:59:54, time: 5.530, data_time: 0.092, loss_rpn_cls: 0.0237, loss_rpn_bbox: 0.0160, loss_cls: 0.3650, acc: 87.1133, loss_bbox: 0.3418, loss: 0.7465, AP50: 0.5480, mAP: 0.5476
Stopping orca context
```