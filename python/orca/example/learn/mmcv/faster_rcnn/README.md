# MMCV Faster-RCNN Training Example

## Prepare the dataset

This training example use the kitti_tiny dataset, you can download it from [here](https://download.openmmlab.com/mmdetection/data/kitti_tiny.zip) and unzip the dataset.

## Prepare the environment

We recommend you to use Anaconda to prepare the environment:

```
conda create -n bigdl-orca pyton=3.7  # "bigdl-orca" is conda environment name, you can use any name you like.
conda activate bigdl-orca

pip install torch torchvision tensorboard
pip install -U openmim
mim install mmcv-full
pip install mmdet

pip install --pre --upgrade bigdl-orca
pip install ray
```

## Download MMDet

In this example, we need to use the config file from the [mmdet repo](https://github.com/open-mmlab/mmdetection).

```
git clone https://github.com/open-mmlab/mmdetection
```

The config file we use in this example is `mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py`

If you want to use the pre-trained checkpoints for further training, also download the checkpoint file

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

`$MMDET_PATH`: your mmdetection repo local path.

The result should looks like this.

```
(MMCVRayEpochRunner pid=188889) 2023-03-13 10:45:51,431 - mmdet - INFO - Start running, host: root@xxx, work_dir: /root/xxx/mmdetection/tutorial_exps
(MMCVRayEpochRunner pid=188889) 2023-03-13 10:45:51,432 - mmdet - INFO - Hooks will be executed in the following order:
(MMCVRayEpochRunner pid=188889) before_run:
(MMCVRayEpochRunner pid=188889) (VERY_HIGH   ) StepLrUpdaterHook
(MMCVRayEpochRunner pid=188889) (NORMAL      ) CheckpointHook
(MMCVRayEpochRunner pid=188889) (LOW         ) DistEvalHook
(MMCVRayEpochRunner pid=188889) (VERY_LOW    ) TextLoggerHook
(MMCVRayEpochRunner pid=188889) (VERY_LOW    ) TensorboardLoggerHook
(MMCVRayEpochRunner pid=188889)  --------------------
(MMCVRayEpochRunner pid=188889) before_train_epoch:
(MMCVRayEpochRunner pid=188889) (VERY_HIGH   ) StepLrUpdaterHook
(MMCVRayEpochRunner pid=188889) (NORMAL      ) NumClassCheckHook
(MMCVRayEpochRunner pid=188889) (NORMAL      ) DistSamplerSeedHook
(MMCVRayEpochRunner pid=188889) (LOW         ) IterTimerHook
(MMCVRayEpochRunner pid=188889) (LOW         ) DistEvalHook
(MMCVRayEpochRunner pid=188889) (VERY_LOW    ) TextLoggerHook
(MMCVRayEpochRunner pid=188889) (VERY_LOW    ) TensorboardLoggerHook
(MMCVRayEpochRunner pid=188889)  --------------------
(MMCVRayEpochRunner pid=188889) before_train_iter:
(MMCVRayEpochRunner pid=188889) (VERY_HIGH   ) StepLrUpdaterHook
(MMCVRayEpochRunner pid=188889) (LOW         ) IterTimerHook
(MMCVRayEpochRunner pid=188889) (LOW         ) DistEvalHook
(MMCVRayEpochRunner pid=188889)  --------------------
(MMCVRayEpochRunner pid=188889) after_train_iter:
(MMCVRayEpochRunner pid=188889) (ABOVE_NORMAL) OptimizerHook
(MMCVRayEpochRunner pid=188889) (NORMAL      ) CheckpointHook
(MMCVRayEpochRunner pid=188889) (LOW         ) IterTimerHook
(MMCVRayEpochRunner pid=188889) (LOW         ) DistEvalHook
(MMCVRayEpochRunner pid=188889) (VERY_LOW    ) TextLoggerHook
(MMCVRayEpochRunner pid=188889) (VERY_LOW    ) TensorboardLoggerHook
(MMCVRayEpochRunner pid=188889)  --------------------
(MMCVRayEpochRunner pid=188889) after_train_epoch:
(MMCVRayEpochRunner pid=188889) (NORMAL      ) CheckpointHook
(MMCVRayEpochRunner pid=188889) (LOW         ) DistEvalHook
(MMCVRayEpochRunner pid=188889) (VERY_LOW    ) TextLoggerHook
(MMCVRayEpochRunner pid=188889) (VERY_LOW    ) TensorboardLoggerHook
(MMCVRayEpochRunner pid=188889)  --------------------
(MMCVRayEpochRunner pid=188889) before_val_epoch:
(MMCVRayEpochRunner pid=188889) (NORMAL      ) NumClassCheckHook
(MMCVRayEpochRunner pid=188889) (NORMAL      ) DistSamplerSeedHook
(MMCVRayEpochRunner pid=188889) (LOW         ) IterTimerHook
(MMCVRayEpochRunner pid=188889) (VERY_LOW    ) TextLoggerHook
(MMCVRayEpochRunner pid=188889) (VERY_LOW    ) TensorboardLoggerHook
(MMCVRayEpochRunner pid=188889)  --------------------
(MMCVRayEpochRunner pid=188889) before_val_iter:
(MMCVRayEpochRunner pid=188889) (LOW         ) IterTimerHook
(MMCVRayEpochRunner pid=188889)  --------------------
(MMCVRayEpochRunner pid=188889) after_val_iter:
(MMCVRayEpochRunner pid=188889) (LOW         ) IterTimerHook
(MMCVRayEpochRunner pid=188889)  --------------------
(MMCVRayEpochRunner pid=188889) after_val_epoch:
(MMCVRayEpochRunner pid=188889) (VERY_LOW    ) TextLoggerHook
(MMCVRayEpochRunner pid=188889) (VERY_LOW    ) TensorboardLoggerHook
(MMCVRayEpochRunner pid=188889)  --------------------
(MMCVRayEpochRunner pid=188889) after_run:
(MMCVRayEpochRunner pid=188889) (VERY_LOW    ) TextLoggerHook
(MMCVRayEpochRunner pid=188889) (VERY_LOW    ) TensorboardLoggerHook
(MMCVRayEpochRunner pid=188889)  --------------------
(MMCVRayEpochRunner pid=188889) 2023-03-13 10:45:51,433 - mmdet - INFO - workflow: [('train', 1)], max: 1 epochs
(MMCVRayEpochRunner pid=188889) 2023-03-13 10:45:51,433 - mmdet - INFO - Checkpoints will be saved to /root/zj/mmdetection/tutorial_exps by HardDiskBackend.
(MMCVRayEpochRunner pid=188889) 2023-03-13 10:46:00,405 - mmcv - INFO - Reducer buckets have been rebuilt in this iteration.
(MMCVRayEpochRunner pid=188889) 2023-03-13 10:46:50,666 - mmdet - INFO - Epoch [1][10/25]       lr: 2.500e-03, eta: 0:01:28, time: 5.918, data_time: 0.222, loss_rpn_cls: 0.0203, loss_rpn_bbox: 0.0190, loss_cls: 0.6038, acc: 78.4180, loss_bbox: 0.4219, loss: 1.0649
(MMCVRayEpochRunner pid=188889) 2023-03-13 10:47:43,460 - mmdet - INFO - Epoch [1][20/25]       lr: 2.500e-03, eta: 0:00:27, time: 5.280, data_time: 0.006, loss_rpn_cls: 0.0312, loss_rpn_bbox: 0.0136, loss_cls: 0.2297, acc: 91.8848, loss_bbox: 0.3163, loss: 0.5908
(MMCVRayEpochRunner pid=188889) 2023-03-13 10:48:09,760 - mmdet - INFO - Saving checkpoint at 1 epochs
(MMCVRayEpochRunner pid=188889) [                                                  ] 0/25, elapsed: 0s, ETA:
[>                                 ] 1/25, 0.3 task/s, elapsed: 4s, ETA:    88s
[>>                                ] 2/25, 0.4 task/s, elapsed: 5s, ETA:    56s
[>>>>                              ] 3/25, 0.5 task/s, elapsed: 6s, ETA:    45s
[>>>>>                             ] 4/25, 0.6 task/s, elapsed: 7s, ETA:    38s
[>>>>>>                            ] 5/25, 0.6 task/s, elapsed: 9s, ETA:    34s
[>>>>>>>                          ] 6/25, 0.6 task/s, elapsed: 10s, ETA:    31s
[>>>>>>>>>                        ] 7/25, 0.6 task/s, elapsed: 11s, ETA:    28s
[>>>>>>>>>>                       ] 8/25, 0.7 task/s, elapsed: 12s, ETA:    26s
[>>>>>>>>>>>                      ] 9/25, 0.7 task/s, elapsed: 13s, ETA:    24s
[>>>>>>>>>>>>                    ] 10/25, 0.7 task/s, elapsed: 15s, ETA:    22s
[>>>>>>>>>>>>>>                  ] 11/25, 0.7 task/s, elapsed: 16s, ETA:    20s
[>>>>>>>>>>>>>>>                 ] 12/25, 0.7 task/s, elapsed: 17s, ETA:    18s
[>>>>>>>>>>>>>>>>                ] 13/25, 0.7 task/s, elapsed: 18s, ETA:    17s
[>>>>>>>>>>>>>>>>>               ] 14/25, 0.7 task/s, elapsed: 19s, ETA:    15s
[>>>>>>>>>>>>>>>>>>>             ] 15/25, 0.7 task/s, elapsed: 20s, ETA:    14s
[>>>>>>>>>>>>>>>>>>>>            ] 16/25, 0.7 task/s, elapsed: 22s, ETA:    12s
[>>>>>>>>>>>>>>>>>>>>>           ] 17/25, 0.7 task/s, elapsed: 23s, ETA:    11s
[>>>>>>>>>>>>>>>>>>>>>>>         ] 18/25, 0.8 task/s, elapsed: 24s, ETA:     9s
[>>>>>>>>>>>>>>>>>>>>>>>>        ] 19/25, 0.8 task/s, elapsed: 25s, ETA:     8s
[>>>>>>>>>>>>>>>>>>>>>>>>>       ] 20/25, 0.8 task/s, elapsed: 26s, ETA:     7s
[>>>>>>>>>>>>>>>>>>>>>>>>>>      ] 21/25, 0.8 task/s, elapsed: 28s, ETA:     5s
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>    ] 22/25, 0.8 task/s, elapsed: 29s, ETA:     4s
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   ] 23/25, 0.8 task/s, elapsed: 30s, ETA:     3s
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  ] 24/25, 0.8 task/s, elapsed: 32s, ETA:     1s
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 25/25, 0.8 task/s, elapsed: 33s, ETA:     0s
(MMCVRayEpochRunner pid=188889)
(MMCVRayEpochRunner pid=188889)
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

- yarn-client mode

If you want to use the pre-trained checkpoints, please first upload the checkpoint file to HDFS, then execute

```bash
python train.py  --dataset /your/dataset/path/to/kitti_tiny/ --cluster_mode yarn-client --config $MMDET_PATH/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py --load_from hdfs://ip:port/your/hdfs/path/to/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth
```

