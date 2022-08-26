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

import torch
import warnings
from typing import Dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.training_type import SingleDevicePlugin, DDPSpawnPlugin
from pytorch_lightning.accelerators.cpu import CPUAccelerator

from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10
from bigdl.nano.common import check_avx512


class CheckIPEXCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        if not trainer.use_ipex:
            warnings.warn("CheckIPEXCallback is used, but ipex is disabled. ") 
            return
        if TORCH_VERSION_LESS_1_10:
            from bigdl.nano.deps.ipex.version_1_9.ipex_torchfunctional import RESTORE_TYPE

            def check_device(obj):
                if torch.is_tensor(obj):
                    if obj.device.type == 'xpu':
                        return True
                    return False
                if isinstance(obj, RESTORE_TYPE):
                    iter_keys = obj.keys() if isinstance(obj, Dict) else range(len(obj))
                    for k in iter_keys:
                        if isinstance(obj[k], RESTORE_TYPE):
                            if not check_device(obj[k]):
                                return False
                return True
            assert check_device(pl_module.state_dict())
        else:
            from intel_extension_for_pytorch.nn.utils._model_convert import _LSTM
            from intel_extension_for_pytorch.nn.utils._weight_prepack import (_IPEXConvNd,
                                                                              _IPEXLinear,
                                                                              _IPEXConvTransposeNd)
            IPEX_LAYERS = (_LSTM, 
                           _IPEXConvNd,
                           _IPEXLinear,
                           _IPEXConvTransposeNd)
            IPEX_ATTR = ('master_weight',
                         'weight_trail',
                         'master_bias',
                         'bias_trail')

            def check_ipex_layers(m):
                if isinstance(m, IPEX_LAYERS):
                    print("model is optimized by IPEX")
                    print(f"model contains layer {m}")
                    return True
                for attr in IPEX_ATTR:
                    if hasattr(m, attr):
                        return True
                for name, sub_m in m.named_children():
                    if check_ipex_layers(sub_m):
                        return True
                return False
            assert check_ipex_layers(pl_module)


class CheckIPEXFusedStepCallback(Callback):
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not check_avx512():
            # IPEX BF16 weight prepack needs the cpu support avx512bw, avx512vl and avx512dq
            return
        if not TORCH_VERSION_LESS_1_10:
            from intel_extension_for_pytorch.optim._optimizer_utils import IPEX_FUSED_OPTIMIZER_LIST
            # IPEX only support one optimizer
            opt = trainer.optimizers[0]
            if type(opt) in IPEX_FUSED_OPTIMIZER_LIST:
                assert opt.fused  # type: ignore
            else:
                # Check non-fused step
                assert hasattr(opt, '_original_step')
                assert getattr(opt, 'step') is not getattr(type(opt), 'step')
