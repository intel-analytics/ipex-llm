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
import os
import subprocess
import sys
import tempfile
from typing import Callable, Optional

import cloudpickle

from . import _worker


def exec_with_worker(func: Callable, *args, env: Optional[dict] = None):
    """
    Call func on subprocess with provided environment variables.

    :param func: a Callable object
    :param args: arguments for the func call
    :param env: a mapping that defines the environment variables for the subprocess
    """
    worker_path = _worker.__file__
    tmp_env = {}
    tmp_env.update(os.environ)
    if env is not None:
        tmp_env.update(env)
    if 'PYTHONPATH' in tmp_env:
        tmp_env['PYTHONPATH'] = ":".join([tmp_env['PYTHONPATH'], *sys.path])
    else:
        tmp_env['PYTHONPATH'] = ":".join(sys.path)
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        param_file = os.path.join(tmp_dir_path, 'param')
        with open(param_file, 'wb') as f:
            cloudpickle.dump([func, *args], f)
        subprocess.run(["python", worker_path, param_file],
                       check=True, env=tmp_env)
        with open(os.path.join(tmp_dir_path, _worker.RETURN_FILENAME), 'rb') as f:
            return cloudpickle.load(f)
