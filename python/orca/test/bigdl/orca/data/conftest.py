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


@pytest.fixture(autouse=True, scope='package')
def orca_context_fixture(request):
    import os
    from bigdl.orca import OrcaContext, init_orca_context, stop_orca_context
    OrcaContext._eager_mode = True
    access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if access_key_id is not None and secret_access_key is not None:
        env = {"AWS_ACCESS_KEY_ID": access_key_id,
               "AWS_SECRET_ACCESS_KEY": secret_access_key}
    else:
        env = None
    sc = init_orca_context(cores=4, spark_log_level="INFO",
                           env=env, object_store_memory="1g",
                           init_ray_on_spark=True)
    yield sc
    stop_orca_context()
