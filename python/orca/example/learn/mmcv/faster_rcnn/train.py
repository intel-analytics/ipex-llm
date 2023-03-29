#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import os

import numpy as np
import os.path as osp
from os.path import exists, join

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.data.file import get_remote_dir_to_local
from bigdl.orca.learn.pytorch.experimential.mmcv.mmcv_ray_estimator import MMCVRayEstimator

import mmcv
from mmcv import Config

from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor


def mmcv_runner_creator(cfg):
    from mmdet.datasets.builder import DATASETS
    from mmdet.datasets.custom import CustomDataset
    from mmdet.models import build_detector
    from mmdet.utils import (compat_cfg, find_latest_checkpoint, get_root_logger)
    from mmdet.core import DistEvalHook, EvalHook, build_optimizer
    from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                             Fp16OptimizerHook, OptimizerHook, build_runner)

    @DATASETS.register_module()
    class KittiTinyDataset(CustomDataset):
        CLASSES = ('Car', 'Pedestrian', 'Cyclist')

        def load_annotations(self, ann_file):
            cat2label = {k: i for i, k in enumerate(self.CLASSES)}
            # load image list from file
            image_list = mmcv.list_from_file(self.ann_file)

            data_infos = []
            # convert annotations to middle format
            for image_id in image_list:
                filename = f'{self.img_prefix}/{image_id}.jpeg'
                image = mmcv.imread(filename)
                height, width = image.shape[:2]

                data_info = dict(filename=f'{image_id}.jpeg', width=width, height=height)

                # load annotations
                label_prefix = self.img_prefix.replace('image_2', 'label_2')
                lines = mmcv.list_from_file(osp.join(label_prefix, f'{image_id}.txt'))

                content = [line.strip().split(' ') for line in lines]
                bbox_names = [x[0] for x in content]
                bboxes = [[float(info) for info in x[4:8]] for x in content]

                gt_bboxes = []
                gt_labels = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []

                # filter 'DontCare'
                for bbox_name, bbox in zip(bbox_names, bboxes):
                    if bbox_name in cat2label:
                        gt_labels.append(cat2label[bbox_name])
                        gt_bboxes.append(bbox)
                    else:
                        gt_labels_ignore.append(-1)
                        gt_bboxes_ignore.append(bbox)

                data_anno = dict(
                    bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                    labels=np.array(gt_labels, dtype=int),
                    bboxes_ignore=np.array(gt_bboxes_ignore,
                                           dtype=np.float32).reshape(-1, 4),
                    labels_ignore=np.array(gt_labels_ignore, dtype=int))

                data_info.update(ann=data_anno)
                data_infos.append(data_info)

            return data_infos

    model = build_detector(cfg.model)

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    distributed = cfg.distributed
    validate = cfg.validate

    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=None))

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=0,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    return runner


def dataloader_creator(cfg):
    distributed = cfg.distributed
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    dataset = build_dataset(cfg.data.train)

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=0,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = build_dataloader(dataset, **train_loader_cfg)
    return data_loaders


def main():
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--dataset', type=str, help='path to dataset')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn-client, yarn-cluster')
    parser.add_argument('--config', type=str, help='The path of config file')
    parser.add_argument("--cores", type=int, default=8,
                        help="The number of cores on each node.")
    parser.add_argument("--num_nodes", type=int, default=1,
                        help="The number of nodes to use.")
    parser.add_argument('--workers_per_node', default=1, type=int,
                        help='The number of torch runners on each node.')
    parser.add_argument('--validate', default=True, type=bool,
                        help='whether do validation during training')
    parser.add_argument('--load_from', default=None, type=str,
                        help='checkpoint path to resume model')
    args = parser.parse_args()

    if args.cluster_mode == "local":
        init_orca_context("local", cores=args.cores, memory="10g")
    elif args.cluster_mode.startswith("yarn"):
        additional = None
        if exists(join(args.dataset, "kitti_tiny.zip")):
            additional = join(args.dataset, "kitti_tiny.zip#", args.dataset)
        init_orca_context(cluster_mode="yarn-client", cores=args.cores, memory="4g",
                          num_nodes=args.num_nodes, driver_cores=4, driver_memory="2g",
                          additional_archive=additional)
    else:
        print("init_orca_context failed. cluster_mode should be one of 'local', 'yarn-client', "
              "but got" + args.cluster_mode)

    cfg = Config.fromfile(args.config)
    cfg.distributed = True
    cfg.validate = args.validate

    # Modify dataset type and path
    cfg.dataset_type = 'KittiTinyDataset'
    cfg.data_root = args.dataset

    cfg.data.test.type = 'KittiTinyDataset'
    cfg.data.test.data_root = args.dataset
    cfg.data.test.ann_file = 'train.txt'
    cfg.data.test.img_prefix = 'training/image_2'

    cfg.data.train.type = 'KittiTinyDataset'
    cfg.data.train.data_root = args.dataset
    cfg.data.train.ann_file = 'train.txt'
    cfg.data.train.img_prefix = 'training/image_2'

    cfg.data.val.type = 'KittiTinyDataset'
    cfg.data.val.data_root = args.dataset
    cfg.data.val.ann_file = 'val.txt'
    cfg.data.val.img_prefix = 'training/image_2'

    # modify num classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = 3

    # If we need to finetune a model based on a pre-trained detector, we need to
    # use load_from to set the path of checkpoints.
    cfg.load_from = args.load_from

    # Set up working dir to save files and logs.
    cfg.work_dir = './tutorial_exps'

    cfg.optimizer.lr = 0.02 / 8
    cfg.lr_config.warmup = None

    cfg.runner.max_epochs = 1
    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 1
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 1
    cfg.log_config.interval = 10

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.device = 'cpu'
    cfg.gpu_ids = [0]

    # We can also use tensorboard to log the training process
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')]

    print(f'Config:\n{cfg.pretty_text}')

    estimator = MMCVRayEstimator(
        mmcv_runner_creator=mmcv_runner_creator,
        config=cfg,
        workers_per_node=args.workers_per_node
    )
    estimator.run([dataloader_creator], cfg.workflow)
    stop_orca_context()


if __name__ == '__main__':
    main()
