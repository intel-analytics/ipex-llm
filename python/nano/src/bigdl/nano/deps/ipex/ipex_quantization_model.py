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

from collections.abc import Sequence
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
from bigdl.nano.pytorch.model import AcceleratedLightningModule
from bigdl.nano.pytorch.context_manager import generate_context_manager
from bigdl.nano.utils.pytorch import patch_attrs_from_model_to_object, jit_convert
from bigdl.nano.utils.common import compare_version
import operator
import torch


class PytorchIPEXQuantizationModel(AcceleratedLightningModule):
    def __init__(self, model: torch.nn.Module, calib_data, q_config=None,
                 input_sample=None, channels_last=None, thread_num=None,
                 from_load=False, inplace=False, jit_strict=True,
                 example_kwarg_inputs=None, enable_onednn=False):
        """
        This is the accelerated model for pytorch and ipex/jit.
        All the external API is based on InferenceOptimizer, so what we have here is
        basically internal APIs and subject to change.

        This PytorchIPEXQuantizationModel will serve for int8 and ipex>1.9 models.
        :param model: the model(nn.module) to be transform if from_load is False
               the accelerated model if from_load is True.
        :param calib_data: calibration data is required for static quantization.
        :param q_config: describes how to quantize a layer or a part of the network
               by providing settings (observer classes) for activations and weights
               respectively. Note that QConfig needs to contain observer classes
               (like MinMaxObserver) or a callable that returns instances on
               invocation, not the concrete observer instances themselves.
               Quantization preparation function will instantiate observers multiple
               times for each of the layers. For more details, please refer
               https://pytorch.org/docs/1.13/generated/torch.quantization.qconfig.
               QConfig.html#torch.quantization.qconfig.QConfig .
        :param input_sample: torch tensor indicate the data sample to be used
               for tracing.
        :param channels_last: if set model and data to be channels-last mode.
        :param thread_num: the thread num allocated for this model.
        :param from_load: this will only be set by _load method.
        :param inplace: whether to perform inplace optimization. Default: ``False``.
        :param jit_strict: Whether recording your mutable container types.
        :param example_kwarg_inputs: keyword arguments of example inputs that will
               be passed to ``torch.jit.trace``. Default to ``None``. Either this
               argument or ``input_sample`` should be specified when ``use_jit`` is
               ``True`` and torch > 2.0, otherwise will be ignored.
        :param enable_onednn: Whether to use PyTorch JIT graph fuser based on oneDNN
               Graph API, which provides a flexible API for aggressive fusion. Default to
               ``False``. For more details, please refer https://github.com/
               pytorch/pytorch/tree/master/torch/csrc/jit/codegen/
               onednn#pytorch---onednn-graph-api-bridge.
        """
        super().__init__(model)
        if from_load:
            self.channels_last = channels_last
            self.jit_strict = jit_strict
            self.enable_onednn = enable_onednn
            self._nano_context_manager = generate_context_manager(accelerator="jit",
                                                                  precision="int8",
                                                                  thread_num=thread_num,
                                                                  enable_onednn=enable_onednn)
            return
        self.channels_last = channels_last
        self.original_state_dict = model.state_dict()
        self.jit_strict = jit_strict
        self.enable_onednn = enable_onednn
        self.original_model = model
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self._nano_context_manager = generate_context_manager(accelerator="jit",
                                                              precision="int8",
                                                              thread_num=thread_num,
                                                              enable_onednn=enable_onednn)
        self.thread_num = thread_num
        if q_config is None:
            # default qconfig
            self.q_config = ipex.quantization.default_static_qconfig
        else:
            self.q_config = q_config

        if input_sample is None:
            input_sample = next(iter(calib_data))
            # data transform : remove y
            if isinstance(input_sample, (tuple, list)) and len(input_sample) > 1:
                input_sample = input_sample[0]
                if self.channels_last:
                    if isinstance(input_sample, torch.Tensor):
                        input_sample = input_sample.to(memory_format=torch.channels_last)
                    else:
                        input_sample = tuple(map(lambda x: x.to(memory_format=torch.channels_last),
                                                 input_sample))

        # prepare model
        self.model = prepare(self.model, self.q_config,
                             example_inputs=input_sample,
                             inplace=inplace)

        # calibration model
        for idx, x in enumerate(calib_data):
            if idx >= calib_data._nano_calib_sample_size:
                break
            # data transform of calib_data
            if isinstance(x, (tuple, list)) and len(x) > 1:
                x = x[0]
            if isinstance(x, Sequence):
                self.model(*x)
            else:
                self.model(x)

        # convert to static quantized model
        self.model = convert(self.model)
        with torch.no_grad():
            self.model = jit_convert(self.model, input_sample,
                                     jit_method='trace',
                                     jit_strict=jit_strict,
                                     example_kwarg_inputs=example_kwarg_inputs)
            self.model = torch.jit.freeze(self.model)
        # patch attributes from original model
        patch_attrs_from_model_to_object(self.original_model, self)

    @property
    def forward_args(self):
        return [input_value.debugName() for input_value in self.model.graph.inputs()
                if not input_value.debugName().startswith('self')]

    def on_forward_start(self, inputs):
        return inputs

    def forward_step(self, *inputs):
        if self.channels_last is True:
            inputs = tuple(map(lambda x: x.to(memory_format=torch.channels_last), inputs))
        return self.model(*inputs)

    def on_forward_end(self, outputs):
        return outputs

    @property
    def status(self):
        status = super().status
        status.update({"channels_last": self.channels_last,
                       "checkpoint": "ckpt.pth",
                       "thread_num": self.thread_num,
                       "jit_strict": self.jit_strict,
                       "enable_onednn": self.enable_onednn})
        return status

    @staticmethod
    def _load(path, model, inplace=False):
        status = PytorchIPEXQuantizationModel._load_status(path)
        checkpoint_path = path / status['checkpoint']
        model = torch.jit.load(checkpoint_path)
        model.eval()
        model = torch.jit.freeze(model)
        from_load = True
        thread_num = None
        if status["thread_num"] is not None and status['thread_num'] != {}:
            thread_num = int(status['thread_num'])
        return PytorchIPEXQuantizationModel(model,
                                            calib_data=None,
                                            channels_last=status['channels_last'],
                                            from_load=from_load,
                                            thread_num=thread_num,
                                            inplace=inplace,
                                            jit_strict=status["jit_strict"],
                                            enable_onednn=status.get('enable_onednn', False))

    def _save_model(self, path, compression="fp32"):
        self.model.save(path / "ckpt.pth")
