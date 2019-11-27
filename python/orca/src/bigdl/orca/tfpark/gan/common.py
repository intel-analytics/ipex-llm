#
# Copyright 2018 Analytics Zoo Authors.
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
from bigdl.optim.optimizer import OptimMethod


class GanOptimMethod(OptimMethod):
    def __init__(self, d_optim, g_optim, g_param_size, d_steps=1, g_steps=1):
        super(GanOptimMethod, self).__init__(None, "float",
                                             d_optim, g_optim, d_steps, g_steps, g_param_size)
