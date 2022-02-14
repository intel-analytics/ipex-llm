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


import tensorflow as tf
from bigdl.nano.automl.hpo.mixin import HPOMixin


class Model(HPOMixin, tf.keras.Model):
    def __init__(
        self,
        model_initor=None,
        model_compiler=None,
        *args,
        **kwargs,
    ):
        # super().__init__(*args, **kwargs)
        if model_initor is None:
            super().__init__(*args, **kwargs)
        else:
            self.model_initor = model_initor
        self.model_compiler = model_compiler

        self.objective = None
        self.study = None
