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

from bigdl.nano.pytorch.trainer import Trainer
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10


class TSTrainer(Trainer):
    @staticmethod
    def optimize(model, accelerator="ipex"):
        '''
        This method helps users to transform their model
        to a model optimized by Intel® Extension for PyTorch.
        The returned model should only be used for inferencing.

        :param model: the pytorch/pytorch-lightning model to be optimized.

        :return: a model optimized by Intel® Extension for PyTorch.
        '''

        import intel_extension_for_pytorch as ipex

        if TORCH_VERSION_LESS_1_10:
            invalidInputError(False,
                              f"`optimize` is only suitable for torch>1.10, ",
                              f"please run `pip install --upgrade torch>=1.10.0` and "
                              f"`pip install intel_extension_for_pytorch`")

        model.eval()
        return ipex.optimize(model)
