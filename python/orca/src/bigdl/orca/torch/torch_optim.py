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
import sys
import io
import torch
from importlib.util import find_spec
from bigdl.orca.torch import zoo_pickle_module
from bigdl.dllib.optim.optimizer import OptimMethod
from bigdl.dllib.utils.log4Error import invalidInputError


if find_spec('jep') is None:
    invalidInputError(False, "jep not found, please install jep first.")


class TorchOptim(OptimMethod):
    """
    TorchOptim wraps a torch optimizer for distributed inference or training.
    """

    def __init__(self, optim_bytes, decayType, bigdl_type="float"):
        """
        :param bigdl_type:
        """
        super(TorchOptim, self).__init__(None, bigdl_type, optim_bytes, decayType)

    @staticmethod
    def from_pytorch(optim, decayType="EpochDecay"):
        """
        :param optim: Pytorch optimizer or LrScheduler.
        :param decayType: one string of EpochDecay, IterationDecay, EpochDecayByScore if
                          the optim is LrScheduler.
                          EpochDecay: call LrScheduler.step() every epoch.
                          IterationDecay: call LrScheduler.step() every iteration.
                          EpochDecayByScore: call LrScheduler.step(val_score) every epoch,
                              val_score is the return value of the first validation method.
                              ReduceLROnPlateau must use this EpochDecayByScore.
        """
        bys = io.BytesIO()
        torch.save(optim, bys, pickle_module=zoo_pickle_module)
        zoo_optim = TorchOptim(bys.getvalue(), decayType)
        return zoo_optim
