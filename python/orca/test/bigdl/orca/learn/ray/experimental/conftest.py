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

import pytest
import os
import tempfile
import shutil
from bigdl.orca import init_orca_context, stop_orca_context

TEMP_WORK_DIR = os.path.join(tempfile.gettempdir(), "mmcv_test_work_dir")


@pytest.fixture(autouse=True, scope='package')
def orca_context_fixture():
    init_orca_context(cores=8, init_ray_on_spark=True)
    yield
    stop_orca_context()
    if os.path.exists(TEMP_WORK_DIR):
        shutil.rmtree(TEMP_WORK_DIR)
