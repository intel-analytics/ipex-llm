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

# MIT License

# Copyright (c) 2021 THUML @ Tsinghua University

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# code adapted from https://github.com/thuml/Autoformer
#

import torch
from bigdl.chronos.forecaster.base_forecaster import BasePytorchForecaster
from bigdl.chronos.model.autoformer import model_creator, 


class AutoformerForecaster(BasePytorchForecaster):
    def __init__(self,
                 seq_len,
                 label_len,
                 pred_len,
                 output_attention,
                 enc_in,
                 freq,
                 dec_in,
                 c_out):
        self.model_config = {
            "seq_len": seq_len,
            "label_len": label_len,
            "pred_len": pred_len,
            "output_attention": output_attention,
            "enc_in": enc_in,
            "freq": freq,
            "dec_in": dec_in,
            "c_out": c_out
        }

        self.internal = AutoFormer()

        super().__init__()

    def fit(self, batch, batch_idx):
        return self.internal.training_step(self, batch, batch_idx)

    def evaluate(self, batch, batch_idx):
        return self.internal.validation_step(self, batch, batch_idx)

    def predict(self, batch, batch_idx):
        return self.internal.predict_step(self, batch, batch_idx)
    
    def configure_optimizers(self):
        return self.internal.configure_optimizers(self)
