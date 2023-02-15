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

from .log4Error import invalidInputError
from .log4Error import invalidOperationError

from .log4Warning import register_suggestion
from .log4Warning import output_suggestions

from .backend import Backend

from .cpuinfo import CPUInfo

from .checker import _inc_checker
from .checker import _ipex_checker
from .checker import _onnxruntime_checker
from .checker import _openvino_checker
from .checker import _bf16_checker
from .checker import _avx512_checker

from .model import AcceleratedModel

from .schedule import schedule_processors

from .subprocess import spawn_new_process

from .version import compare_version

from .inspect import get_default_args

from .decorator import deprecated

from .affinity_core import get_affinity_core_num

from .env import _env_variable_is_set
from .env import _find_library

from .optimizer import *
