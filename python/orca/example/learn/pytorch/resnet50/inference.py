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
# ==============================================================================
# Most of the pytorch code is adapted from:
# https://github.com/IntelAI/models/blob/master/models/image_recognition/
# pytorch/common/main.py
#

import os
import argparse

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from bigdl.dllib.utils.log4Error import *
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import TrainingOperator

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument("--cluster_mode", type=str, default="local",
                    help="The cluster mode, such as local, yarn or standalone.")
parser.add_argument("--master", type=str, default=None,
                    help="The master url, only used when cluster mode is standalone.")
parser.add_argument("--cores", type=int, default=4,
                    help="The number of cores on each node.")
parser.add_argument("--num_nodes", type=int, default=1,
                    help="The number of nodes to use.")
parser.add_argument('--workers_per_node', default=1, type=int,
                    help='The number of torch runners on each node.')
parser.add_argument('--ipex', action='store_true', default=False,
                    help='use intel pytorch extension')
parser.add_argument('--jit', action='store_true', default=False,
                    help='enable ipex jit fusionpath')
parser.add_argument('--int8', action='store_true', default=False,
                    help='enable ipex int8 path')
parser.add_argument('--bf16', action='store_true', default=False,
                    help='enable ipex bf16 path')
parser.add_argument('--bf32', action='store_true', default=False,
                    help='enable ipex bf32 path')
parser.add_argument('-b', '--batch_size', default=256, type=int)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument("--dummy", action='store_true',
                    help="using dummy data to test the performance of inference")
parser.add_argument('--hub', action='store_true', default=False,
                    help='use model with torch hub')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--steps', default=-1, type=int,
                    help='steps for validation')
parser.add_argument('--calibration', action='store_true', default=False,
                    help='doing calibration step for int8 path')
parser.add_argument('--configure_dir', default='configure.json', type=str, metavar='PATH',
                    help='path to int8 configures, default file name is configure.json')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')


class ResNetPerfOperator(TrainingOperator):
    def validate(self, val_iterator, info, metrics, num_steps=None):
        self.model.eval()
        from bigdl.orca.learn.metrics import Metric
        metrics = Metric.convert_metrics_dict(metrics, backend="pytorch")
        losses = []
        total_samples = 0

        if self.config["ipex"] and self.config["int8"] and self.config["calibration"]:
            print("running int8 calibration step\n")
            import intel_extension_for_pytorch as ipex
            from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
            qconfig = QConfig(
                activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric,
                                                    dtype=torch.qint8),
                weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8,
                                                          qscheme=torch.per_channel_symmetric))
            x = torch.randn(1, 3, 224, 224)
            prepared_model = ipex.quantization.prepare(self.model, qconfig, x, inplace=True)
            with torch.no_grad():
                for i, (images, target) in enumerate(val_iterator):
                    images = images.contiguous(memory_format=torch.channels_last)
                    prepared_model(images)
                    if i == 4:
                        print(i)
                        break
                prepared_model.save_qconf_summary(self.config["configure_dir"])
                print(".........calibration step done..........")

        if not num_steps:
            num_steps = len(val_iterator)

        if self.use_tqdm and self.world_rank == 0:
            from tqdm import tqdm
            _progress_bar = tqdm(
                total=num_steps,
                unit="batch",
                leave=False)

        # TODO: support warmup
        with torch.no_grad():
            if self.config["dummy"]:
                images = torch.randn(self.config["batch"], 3, 224, 224)
                target = torch.arange(1, self.config["batch"] + 1).long()
                batch = images, target
            else:
                batch = None
            for batch_idx in range(num_steps):
                batch_info = {"batch_idx": batch_idx}
                batch_info.update(info)
                if not self.config["dummy"]:
                    batch = next(val_iterator)
                output, target, loss = self.forward_batch(batch, batch_info)
                if self.use_tqdm and self.world_rank == 0:
                    _progress_bar.n = batch_idx + 1
                    postfix = {}
                    # postfix.update(loss=loss)
                    _progress_bar.set_postfix(postfix)
                from bigdl.orca.learn.pytorch.utils import get_batchsize
                num_samples = get_batchsize(target)
                total_samples += num_samples
                losses.append(loss.item() * num_samples)
                for metric in metrics.values():
                    metric(output, target)

        result = {name: metric.compute() for name, metric in metrics.items()}
        result["val_loss"] = sum(losses) / total_samples
        result["num_samples"] = total_samples
        return result

    def forward_batch(self, batch, batch_info):
        images, target = batch
        if self.config["ipex"]:
            images = images.contiguous(memory_format=torch.channels_last)
        if self.config["bf16"]:
            images = images.to(torch.bfloat16)

        # compute output
        with self.timers.record("eval_fwd"):
            if not self.config["jit"] and self.config["bf16"]:
                with torch.cpu.amp.autocast():
                    output = self.model(images)
            else:
                output = self.model(images)

        if self.config["bf16"]:
            output = output.to(torch.float32)
        loss = self.criterion(output, target)

        return output, target, loss


def main():
    args = parser.parse_args()
    print(args)

    if args.jit and args.int8:
        invalidInputError(False, "jit path is not available for int8 path using ipex")
    if not args.ipex:
        # for official pytorch, int8 and jit path is not enabled.
        invalidInputError(not args.int8, "int8 path is not enabled for official pytorch")
        invalidInputError(not args.jit, "jit path is not enabled for official pytorch")

    env = {"MALLOC_CONF": "oversize_threshold:1,background_thread:true,metadata_thp:"
                          "auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000",
           "DNNL_PRIMITIVE_CACHE_CAPACITY": "1024",
           "KMP_BLOCKTIME": "1",
           "KMP_AFFINITY": "granularity=fine,compact,1,0"}
    if "LD_PRELOAD" in os.environ:
        env["LD_PRELOAD"] = os.environ["LD_PRELOAD"]

    if args.cluster_mode == "local":
        init_orca_context("local", cores=args.cores, memory="10g")
    elif args.cluster_mode == "standalone":
        init_orca_context("standalone", master=args.master,
                          cores=args.cores, num_nodes=args.num_nodes,
                          memory="10g", driver_cores=1, driver_memory="2g", env=env)
    elif args.cluster_mode == "yarn":
        init_orca_context("yarn-client", cores=args.cores,
                          num_nodes=args.num_nodes, memory="10g",
                          driver_cores=4, driver_memory="2g", env=env)
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn' and 'standalone', "
                          "but got " + args.cluster_mode)
    validate(args)
    stop_orca_context()


def validate(args):

    def val_loader_func(config, batch_size):
        # Data loading code
        invalidInputError(args.data is not None,
                          "please set dataset path if you want to using real data")
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        return val_loader

    def model_creator(config):
        if args.seed is not None:
            import random
            random.seed(args.seed)
            torch.manual_seed(args.seed)
        arch = 'resnet50'
        if args.hub:
            torch.set_flush_denormal(True)
            model = torch.hub.load('facebookresearch/WSL-Images', arch)
        else:
            # create model
            if args.pretrained:
                print("=> using pre-trained model '{}'".format(arch))
                model = models.__dict__[arch](pretrained=True)
            else:
                print("=> creating model '{}'".format(arch))
                model = models.__dict__[arch]()

        if args.ipex:
            model.eval()
            print("using ipex model to do inference\n")
            import intel_extension_for_pytorch as ipex

            if args.int8:
                if not args.calibration:
                    from torch.ao.quantization import MinMaxObserver, \
                        PerChannelMinMaxObserver, QConfig
                    x = torch.randn(args.batch_size, 3, 224, 224) \
                        .contiguous(memory_format=torch.channels_last)
                    qconfig = QConfig(
                        activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric,
                                                            dtype=torch.qint8),
                        weight=PerChannelMinMaxObserver.with_args(
                            dtype=torch.qint8,
                            qscheme=torch.per_channel_symmetric))
                    prepared_model = ipex.quantization.prepare(model, qconfig, x, inplace=True)
                    prepared_model.load_qconf_summary(qconf_summary=args.configure_dir)
                    model = ipex.quantization.convert(prepared_model)
                    model = torch.jit.trace(model, x)
                    model = torch.jit.freeze(model.eval())
                    y = model(x)
                    y = model(x)
                    print("running int8 evaluation step\n")
            else:
                # for ipex path, always convert model to channels_last for bf16, fp32.
                # TODO: int8 path: https://jira.devtools.intel.com/browse/MFDNN-6103
                model = model.to(memory_format=torch.channels_last)

                if args.bf32:
                    ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.BF32, device="cpu")
                    print("using bf32 fmath mode\n")
                if args.bf16:
                    model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
                    print("running bfloat16 evaluation step\n")
                else:
                    model = ipex.optimize(model, dtype=torch.float32, inplace=True)
                    print("running fp32 evaluation step\n")

                if args.jit:
                    x = torch.randn(args.batch_size, 3, 224, 224) \
                        .contiguous(memory_format=torch.channels_last)
                    if args.bf16:
                        x = x.to(torch.bfloat16)
                        with torch.cpu.amp.autocast(), torch.no_grad():
                            model = torch.jit.trace(model, x).eval()
                    else:
                        with torch.no_grad():
                            model = torch.jit.trace(model, x).eval()
                    model = torch.jit.freeze(model)
        else:
            print("using official pytorch model to do inference\n")

        model.eval()
        return model

    def optimizer_creator(model, config):
        return None

    if args.dummy:
        number_iter = args.steps if args.steps > 0 else 200
    else:
        number_iter = args.steps if args.steps > 0 else None
    if args.calibration:
        number_iter = 100

    from bigdl.orca.learn.pytorch import Estimator
    from bigdl.orca.learn.metrics import Accuracy

    config = vars(args).copy()
    batch = config.pop("batch_size")
    config["batch"] = batch  # Dummy data needs batch_size in TrainingOperator
    backend = "ray"
    est = Estimator.from_torch(model=model_creator,
                               optimizer=optimizer_creator,
                               loss=nn.CrossEntropyLoss(),
                               metrics=[Accuracy()],
                               backend=backend,
                               config=config,
                               workers_per_node=args.workers_per_node,
                               training_operator_cls=ResNetPerfOperator,
                               use_tqdm=True)

    result = est.evaluate(data=val_loader_func, batch_size=args.batch_size,
                          num_steps=number_iter, profile=True)
    for r in result:
        print("{}: {}".format(r, result[r]))

    print('---------')
    print('total number of records:', result['num_samples'])
    print('batch_size for each worker:', args.batch_size)
    num_samples_per_worker = result['num_samples'] / (args.workers_per_node * args.num_nodes)
    print('num_samples for each worker: around', num_samples_per_worker)
    print('num_batches for each worker: around', num_samples_per_worker // args.batch_size)
    mean_validation_s = result['profile']['mean_validation_s']
    mean_eval_fwd_s = result['profile']['mean_eval_fwd_s']
    print('avg_val_time for each worker:', mean_validation_s)
    print('avg_forward_time for each batch:', mean_eval_fwd_s)
    latency = mean_eval_fwd_s / args.batch_size * 1000
    perf = args.batch_size / mean_eval_fwd_s
    print('inference latency %.3f ms' % latency)
    print("Throughput: {:.3f} fps".format(perf))
    print("Accuracy: {top1:.3f} ".format(top1=result["Accuracy"]))

if __name__ == '__main__':
    main()
