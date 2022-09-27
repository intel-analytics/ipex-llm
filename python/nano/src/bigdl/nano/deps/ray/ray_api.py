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


def create_ray_multiprocessing_backend():
    from bigdl.nano.deps.ray.ray_backend import RayBackend
    return RayBackend()


def create_RayStrategy(*args, **kwargs):
    """Create ray strategy."""
    from .ray_distributed import RayStrategy
    return RayStrategy(*args, **kwargs)


# this method is only for API compatibility when using pl 1.4,
# and will be removed soon
def distributed_ray(*args, **kwargs):
    from bigdl.nano.utils.log4Error import invalidInputError
    invalidInputError(False, "bigdl-nano no longer support ray backend "
                      "when using pytorch lightning 1.4, please upgrade your "
                      "pytorch lightning to 1.6.4")
