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

from typing import Any

from pytorch_lightning.accelerators.accelerator import Accelerator

from bigdl.nano.common import check_avx512


class IPEXAccelerator(Accelerator):
    """Accelerator for XPU devices."""

    @staticmethod
    def is_available() -> bool:
        """Detect if IPEX accelerator is available."""
        if not check_avx512():
            Warning("Enable ipex in a cpu instruction set "
                    "without avx512 may cause some random error. "
                    "Fall back to cpu device.")
            return False
        return True

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return 1

    @staticmethod
    def parse_devices(devices: Any) -> Any:
        """Accelerator device parsing logic."""
        pass

    @staticmethod
    def get_parallel_devices(devices: Any) -> Any:
        """Gets parallel devices for the Accelerator."""
        pass
