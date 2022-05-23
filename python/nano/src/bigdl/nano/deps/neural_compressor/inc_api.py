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
from bigdl.nano.utils.log4Error import invalidInputError


def QuantizationINC(framework: str,
                    conf='',
                    approach='post_training_static_quant',
                    tuning_strategy='bayesian',
                    accuracy_criterion: dict = {'relative': 0.99, 'higher_is_better': True},
                    timeout=0,
                    max_trials=1,
                    inputs=None,
                    outputs=None):
    from .core import QuantizationINC as Quantization
    return Quantization(framework, conf, approach, tuning_strategy, accuracy_criterion,
                        timeout, max_trials, inputs, outputs)


def check_pytorch_dataloaders(model, loaders):
    from .pytorch.dataloader import check_loaders
    return check_loaders(model, loaders)


def tf_dataset_to_inc_dataloader(tf_dataset, batchsize):
    from neural_compressor.experimental import common
    return common.DataLoader(tf_dataset, batchsize)


def PytorchQuantizedModel(model):
    from .pytorch.quantized_model import PytorchQuantizedModel
    return PytorchQuantizedModel(model)


def load_inc_model(path, model, framework):
    if framework == 'pytorch':
        from .pytorch.quantized_model import PytorchQuantizedModel
        return PytorchQuantizedModel._load(path, model)
    elif framework == 'tensorflow':
        invalidInputError(False, "QuantizedTensorflowModel loading is not implemented yet.")
    else:
        invalidInputError(False,
                          "The value {} for framework is not supported."
                          " Please choose from 'pytorch'/'tensorflow'.")
