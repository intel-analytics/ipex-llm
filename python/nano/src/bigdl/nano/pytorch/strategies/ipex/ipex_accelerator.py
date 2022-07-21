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


from pytorch_lightning.accelerators.accelerator import Accelerator
from typing import Any


class IPEXAccelerator(Accelerator):
    """IPEX accelerator."""

    @staticmethod
    def is_available() -> bool:
        """Detect if IPEX accelerator is available."""
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
