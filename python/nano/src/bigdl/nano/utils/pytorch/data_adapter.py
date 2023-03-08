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

from typing import Optional, Sequence

import torch
from bigdl.nano.utils.common import invalidInputError
from torch.utils.data import DataLoader

from .dataset import RepeatDataset, remove_batch_dim_fn
from .model_info import ModelInfo


class DataAdapter:
    accelerator_list = [None, 'jit', 'openvino', 'onnxruntime']
    precision_list = ['fp32', 'int8', 'fp16', 'bf16']

    def __init__(self, model):
        self.model = model
        self.model_info = ModelInfo(model)

    def get_data(self, input_data: Optional = None, accelerator: Optional[str] = None,
                 precision: Optional[str] = 'fp32'):
        invalidInputError(accelerator in self.accelerator_list,
                          f'accelerator should be one of {self.accelerator_list}, '
                          f'but get {accelerator}.')
        invalidInputError(precision in self.precision_list,
                          f'precision should be one of {self.precision_list}, '
                          f'but get {precision}.')

        if isinstance(input_data, DataLoader):
            input_sample, input_label = self._split_input_label(next(iter(input_data)))
        elif isinstance(input_data, Sequence):
            input_sample, input_label = self._split_input_label(input_data)
        else:
            input_sample = input_data
            input_label = torch.ones(1)

        if input_sample is None:
            input_sample = self.get_input_example_from_model()
            input_label = torch.ones(1)
        self.model.example_input_array = input_sample
        return input_sample, input_label

    def _split_input_label(self, input_data):
        if isinstance(input_data, Sequence) and len(input_data) > 0:
            input_data_length = len(input_data)
            if input_data_length == 1:
                input_sample = input_data[0]
                input_label = torch.ones(1)
            elif len(input_data) == 2:
                input_sample = input_data[0]
                input_label = input_data[1]
            else:
                input_sample = tuple(input_data[:len(self.model_info.forward_args)])
                input_label = tuple(input_data[len(self.model_info.forward_args):])
        else:
            input_sample = input_data
            input_label = torch.ones(1)
        return input_sample, input_label

    def get_dataloader(self, input_data: Optional = None, accelerator: Optional[str] = None,
                       precision: Optional[str] = 'fp32'):
        input_sample, input_label = self.get_data(input_data, accelerator)
        # turn training_data into dataset
        dataset = RepeatDataset(sample=(input_sample, input_label), num=1)
        dataloader = DataLoader(dataset, batch_size=1)
        dataloader = remove_batch_dim_fn(dataloader)
        return dataloader

    def get_input_example_from_model(self):
        """
        This function will search all dataloaders parameters in model.
        Try to generate an input sample.
        """
        input_sample = None
        if getattr(self.model, "example_input_array", None) is not None:
            input_sample = self.model.example_input_array
        elif getattr(self.model, "trainer", None):
            # the default implementation of model.test_dataloader/train_dataloader/val_dataloader
            # /predict_dataloader will throw an exception in pytorch lightning 1.6
            for dataloader_fn in [self.model.test_dataloader, self.model.train_dataloader,
                                  self.model.val_dataloader]:
                try:
                    dataloader = dataloader_fn()
                    input_sample = next(iter(dataloader))
                    if isinstance(input_sample, Sequence):
                        input_sample = tuple(list(input_sample)[:len(self.model_info.forward_args)])
                    break
                except Exception as _e:
                    pass

            if input_sample is None:
                try:
                    predict_dataloader = self.model.predict_dataloader()
                    input_sample = tuple(next(iter(predict_dataloader)))
                except Exception as _e:
                    pass
        else:
            invalidInputError(False,
                              "You must specify an input_sample or call `Trainer.fit` "
                              "on the model first to use `eval_openvino`")
        return input_sample

    def complement_input_sample(self, input_sample):
        """
        This function will give a complemented input sample
        Mainly using default value to complete.
        """
        input_sample_length = 1
        if isinstance(input_sample, Sequence):
            input_sample_length = len(input_sample)

        # check if input_sample need complement
        if len(self.model_info.forward_args) == input_sample_length:
            return input_sample

        # check if more input sample should be provided
        if len(self.model_info.forward_args) > \
                len(self.model_info.forward_defaults) + input_sample_length:
            invalidInputError(False, "not enough input_sample provided!")

        # complement the input sample by defaults
        if isinstance(input_sample, Sequence):
            input_sample_complement = input_sample
            input_sample_complement += \
                self.model_info.forward_defaults[-(len(
                    self.model_info.forward_args) - input_sample_length):]
        else:
            input_sample_complement = [input_sample]
            input_sample_complement += \
                list(self.model_info.forward_defaults[-(len(
                    self.model_info.forward_args) - input_sample_length):])

        return tuple(input_sample_complement)
