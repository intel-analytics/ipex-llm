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

# This file is adapted from ray_lightning. https://github.com/ray-project/ray_lightning
# Copyright 2021 Ray Lightning Author
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


from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.utilities import rank_zero_only


class RayEnvironment(ClusterEnvironment):
    """Environment for PTL training on a Ray cluster."""

    def __init__(self, world_size):
        self.set_world_size(world_size)
        self._global_rank = 0
        self._is_remote = False

    def creates_children(self) -> bool:
        return False

    def master_address(self) -> str:
        raise NotImplementedError

    def master_port(self) -> int:
        raise NotImplementedError

    def world_size(self) -> int:
        return self._world_size

    def set_world_size(self, size: int) -> None:
        self._world_size = size

    def global_rank(self) -> int:
        return self._global_rank

    def set_global_rank(self, rank: int) -> None:
        self._global_rank = rank
        rank_zero_only.rank = rank  # type: ignore

    def set_remote_execution(self, is_remote: bool) -> None:
        self._is_remote = is_remote

    def is_remote(self) -> bool:
        return self._is_remote

    def local_rank(self) -> int:
        raise NotImplementedError

    def node_rank(self) -> int:
        raise NotImplementedError

    @property
    def creates_processes_externally(self) -> bool:
        """Whether the environment creates the subprocesses or not."""

    @property
    def main_address(self) -> str:
        """The main address through which all processes connect and communicate."""

    @property
    def main_port(self) -> int:
        """An open and configured port in the main node through which all processes communicate."""

    @staticmethod
    def detect() -> bool:
        """Detects the environment settings corresponding to this cluster \
        and returns ``True`` if they match."""
