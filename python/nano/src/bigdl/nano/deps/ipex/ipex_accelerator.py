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
from pytorch_lightning.accelerators.accelerator import Accelerator
from bigdl.nano.common import check_avx512
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class IPEXAccelerator(Accelerator):
    """ Accelerator for XPU devices. """

    def setup_environment(self, root_device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not CPU.
        """
        super().setup_environment(root_device)
        if root_device.type != "cpu":
            raise MisconfigurationException(f"Device should be CPU, got {root_device} instead.")

    def is_available() -> bool:
        '''
        return: if IPEX accelerator is available
        '''
        # TODO: no device check api available so far, just check instruction set
        if not check_avx512():
            warning("Enable ipex in a cpu instruction set"
                    " without avx512 may cause some random error."
                    "Fall back to cpu device.")
            return False
