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

from bigdl.nano.utils.pytorch import generate_channels_last_available,\
    apply_proper_channels_last, transform_state_dict_to_dtype, \
    patch_attrs_from_model_to_object, jit_convert
from bigdl.nano.pytorch.model import AcceleratedLightningModule
from bigdl.nano.pytorch.context_manager import generate_context_manager
from bigdl.nano.deps.ipex.ipex_api import ipex_optimize
from bigdl.nano.utils.common import invalidInputError, compare_version
import operator
import torch
import copy


class PytorchIPEXJITModel(AcceleratedLightningModule):
    def __init__(self, model: torch.nn.Module, input_sample=None, use_ipex=False, dtype=None,
                 use_jit=False, channels_last=None, channels_last_available=[],
                 thread_num=None, from_load=False, inplace=False, jit_strict=True,
                 jit_method=None, weights_prepack=None, enable_onednn=True,
                 compression="fp32", example_kwarg_inputs=None):
        """
        This is the accelerated model for pytorch and ipex/jit.
        All the external API is based on InferenceOptimizer, so what we have here is
        basically internal APIs and subject to change.

        This PytorchIPEXJITModel will serve for fp32 and ipex>1.9 models.
        :param model: the model(nn.module) to be transform if from_load is False
               the accelerated model if from_load is True.
        :param input_sample: torch tensor indicate the data sample to be used
               for tracing.
        :param use_ipex: if use ipex to optimize the model
        :param dtype (torch.dtype): Only works for ``torch.bfloat16``. Model parameters
                                    will be casted to ``torch.bfloat16`` if dtype is set
                                    to ``torch.bfloat16``. The default value is None,
                                    meaning do nothing.
        :param use_jit: if use jit to accelerate the model
        :param channels_last: if set model and data to be channels-last mode.
        :param channels_last_available: only passed by _load method,
                                        to decide which input can be converted
                                        to channels-last mode.
        :param thread_num: the thread num allocated for this model.
        :param from_load: this will only be set by _load method.
        :param inplace: whether to perform inplace optimization. Default: ``False``.
        :param jit_strict: Whether recording your mutable container types.
        :param jit_method: use ``jit.trace`` or ``jit.script`` to
               convert a model to TorchScript.
        :param weights_prepack: Whether to perform weight prepack for convolution and linear
               to avoid oneDNN weights reorder. The default value is None. Explicitly setting
               this knob overwrites the configuration set by level knob. Only valid when
               ``use_ipex=True``, otherwise will be ignored.
        :param enable_onednn: Whether to use PyTorch JIT graph fuser based on oneDNN Graph
               API, which provides a flexible API for aggressive fusion. Default to
               ``True``, only valid when use_jit is ``True``, otherwise will be ignored.
        :param compression: str. This parameter only effective for jit, ipex or pure
               pytorch model with fp32 or bf16 precision. Defaultly, all models are saved
               by dtype=fp32 for their parameters. If users set a lower precision, a smaller
               file sill be saved with some accuracy loss. Users always need to use nano
               to load the compressed file if compression is set other than "fp32".
               Currently, "bf16" and "fp32"(default) are supported.
        :param example_kwarg_inputs: keyword arguments of example inputs that will be passed
               to ``torch.jit.trace``. Default to ``None``. Either this argument or
               ``input_sample`` should be specified when ``use_jit`` is ``True`` and
               torch > 2.0, otherwise will be ignored.
        """
        super().__init__(model)
        if from_load:
            self.use_ipex = use_ipex
            self.use_jit = use_jit
            self.channels_last = channels_last
            self.jit_strict = jit_strict
            self.jit_method = jit_method
            self.weights_prepack = weights_prepack
            self.compression = compression
            if self.channels_last:
                try:
                    self.model = self.model.to(memory_format=torch.channels_last)
                except Exception as _e:
                    self.model = self.model.to(memory_format=torch.channels_last_3d)
                self.channels_last_available = channels_last_available
            self.enable_onednn = enable_onednn
            _accelerator = "jit" if use_jit is True else None
            self._nano_context_manager = generate_context_manager(accelerator=_accelerator,
                                                                  precision="fp32",
                                                                  thread_num=thread_num,
                                                                  enable_onednn=enable_onednn)
            return
        self.channels_last = channels_last
        self.original_state_dict = model.state_dict()
        self.use_ipex = use_ipex
        self.use_jit = use_jit
        self.jit_strict = jit_strict
        self.jit_method = jit_method
        self.weights_prepack = weights_prepack
        self.compression = compression
        self.original_model = model
        self.input_sample = input_sample
        if self.channels_last:
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception as _e:
                self.model = self.model.to(memory_format=torch.channels_last_3d)
            if channels_last_available:  # init from _load, the channels_last_available is not none
                self.channels_last_available = channels_last_available
            else:
                self.channels_last_available = generate_channels_last_available(input_sample)
        else:
            self.channels_last_available = []
        if self.use_ipex:
            self.model = ipex_optimize(self.model, dtype=dtype, inplace=inplace,
                                       weights_prepack=weights_prepack)

        if self.use_jit:
            with torch.no_grad():
                with torch.cpu.amp.autocast(enabled=dtype == torch.bfloat16):
                    self.model = jit_convert(self.model, input_sample,
                                             jit_method=jit_method,
                                             jit_strict=jit_strict,
                                             example_kwarg_inputs=example_kwarg_inputs)
                    if dtype != torch.bfloat16 or self.use_ipex:
                        self.model = torch.jit.freeze(self.model)
        _accelerator = "jit" if use_jit is True else None
        self._nano_context_manager = generate_context_manager(accelerator=_accelerator,
                                                              precision="fp32",
                                                              thread_num=thread_num,
                                                              enable_onednn=enable_onednn)
        self.thread_num = thread_num
        self.enable_onednn = enable_onednn
        # patch attributes from original model
        patch_attrs_from_model_to_object(self.original_model, self)

    @property
    def forward_args(self):
        return [input_value.debugName() for input_value in self.model.graph.inputs()
                if not input_value.debugName().startswith('self')]

    def forward(self, *inputs, **kwargs):
        if self.channels_last:
            # generate channels_last_available list is possible
            # this won't affect inference latency much since it will only run 1 time
            if not self.channels_last_available:
                self.channels_last_available = generate_channels_last_available(inputs)

            # change the data to suitable mem format
            converted_input_length = min(len(self.channels_last_available), len(inputs))
            inputs = tuple(map(
                lambda idx: apply_proper_channels_last(
                    self.channels_last_available[idx], inputs[idx]),
                range(converted_input_length))) + inputs[converted_input_length:]

        return self.model(*inputs, **kwargs)

    @property
    def status(self):
        status = super().status
        status.update({"use_ipex": self.use_ipex,
                       "use_jit": self.use_jit,
                       "channels_last": self.channels_last,
                       "channels_last_available": self.channels_last_available,
                       "checkpoint": "ckpt.pth",
                       "thread_num": self.thread_num,
                       "jit_strict": self.jit_strict,
                       "jit_method": self.jit_method,
                       "weights_prepack": self.weights_prepack,
                       "enable_onednn": self.enable_onednn,
                       "compression": self.compression})
        return status

    @staticmethod
    def _load(path, model, input_sample=None, inplace=False):
        status = PytorchIPEXJITModel._load_status(path)
        if isinstance(path, dict):
            checkpoint_path = path[status['checkpoint']]
        else:
            checkpoint_path = path / status['checkpoint']
        if status["use_jit"]:
            if status['compression'] == "bf16":
                invalidInputError(model is not None,
                                  "You must pass model when loading this model "
                                  "which was saved with compression precision.")
                invalidInputError(input_sample is not None,
                                  "You must pass input_sample when loading this model "
                                  "which was saved with compression precision.")
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                if status['compression'] == "bf16":
                    state_dict = transform_state_dict_to_dtype(state_dict, dtype="fp32")
                if inplace is False:
                    model = copy.deepcopy(model)
                model.load_state_dict(state_dict)
                from_load = False
            else:
                # for fp32, normal jit loading
                if status["use_ipex"]:
                    import intel_extension_for_pytorch as ipex
                model = torch.jit.load(checkpoint_path)
                model.eval()
                model = torch.jit.freeze(model)
                from_load = True
        else:
            state_dict = torch.load(checkpoint_path)
            model.eval()
            if status['compression'] == "bf16":
                state_dict = transform_state_dict_to_dtype(state_dict, dtype="fp32")
            model.load_state_dict(state_dict)
            from_load = False
        thread_num = status.get('thread_num', None)
        if thread_num == {}:
            thread_num = None
        if thread_num is not None:
            thread_num = int(status['thread_num'])
        return PytorchIPEXJITModel(model,
                                   input_sample=input_sample,
                                   use_ipex=status['use_ipex'],
                                   use_jit=status['use_jit'],
                                   channels_last=status['channels_last'],
                                   channels_last_available=status.get('channels_last_available',
                                                                      None),
                                   from_load=from_load,
                                   thread_num=thread_num,
                                   inplace=inplace,
                                   jit_strict=status.get('jit_strict', True),
                                   jit_method=status.get('jit_method', None),
                                   weights_prepack=status.get('weights_prepack', None),
                                   enable_onednn=status.get('enable_onednn', False),
                                   compression=status.get('compression', "fp32"),)

    def _save_model(self, path, compression="fp32"):
        if self.use_jit:
            if compression == "bf16":
                # for jit, if we want to compress its precision at saving,
                # we need to save original model's state dict with compression precision
                self.compression = "bf16"
                bf16_sd = transform_state_dict_to_dtype(self.original_state_dict, dtype="bf16")
                torch.save(bf16_sd, path / "ckpt.pth")
            elif compression == "fp32":
                # normal torch.jit.save for fp32 model
                self.compression = "fp32"
                self.model.save(path / "ckpt.pth")
            else:
                invalidInputError(False,
                                  "compression does not support {} precision for jit accelerator "
                                  "fow now.".format(compression))
        else:
            if compression == "bf16":
                self.compression = "bf16"
                bf16_sd = transform_state_dict_to_dtype(self.original_state_dict, dtype="bf16")
                torch.save(bf16_sd, path / "ckpt.pth")
            else:
                self.compression = "fp32"
                torch.save(self.original_state_dict, path / "ckpt.pth")
