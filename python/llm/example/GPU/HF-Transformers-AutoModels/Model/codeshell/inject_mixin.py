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
# This file is adapted from
# https://github.com/tomaarsen/attention_sinks/blob/main/attention_sinks/inject_mixin.py
#

import types
from typing import Callable, Optional

from transformers.utils import logging

from attention_sinks.attention_sink_kv_cache import AttentionSinkKVCache
from attention_sinks.generation.utils import _update_model_kwargs_for_generation

logger = logging.get_logger(__name__)

MODEL_NAME_MAPPING = {
    "codeshell": "CodeShellModel"
}
ATTENTION_NAME_MAPPING = {
    "codeshell": "CodeShellAttention"
}
KV_DIM_MAPPING = {
    "codeshell": (1, 1)
}

class InjectAttentionSinksMixin:
    @classmethod
    def from_pretrained(cls, model, *model_args, **kwargs):
        # Separate Attention Sink kwargs from regular kwargs
        attention_sink_kwargs = {key: value for key, value in kwargs.items() if key.startswith("attention_sink")}
        for key in attention_sink_kwargs:
            kwargs.pop(key)
        
        model_type = model.config.model_type
        if model_type not in MODEL_NAME_MAPPING:
            raise NotImplementedError(
                f"`attention_sinks` does not support models with the `{model_type}` architecture at this time."
            )

        # Enable position shifting attention
        call_count = cls._inject_pos_shift_attention(model)
        if call_count is not None:
            logger.warn(
                f"[Attention Sinks] Injected Position Shifting into {call_count} attention class{'es' if call_count != 1 else ''}."
            )

        # Inject the Attention Sink KV Cache to the model
        call_count = cls._inject_attention_sink_kv_cache(model, **attention_sink_kwargs)
        logger.warn(
            f"[Attention Sinks] Injected Attention Sink KV Cache into {call_count} model class{'es' if call_count != 1 else ''}."
        )

        # Overwrite broken model kwargs, prevents indexing error when generating
        # The default _update_model_kwargs_for_generation expects the seq_length to keep growing
        # as generation occurs, but that isn't the case
        model._update_model_kwargs_for_generation = types.MethodType(_update_model_kwargs_for_generation, model)

        return model

    @classmethod
    def _inject_pos_shift_attention(cls, model) -> Optional[int]:
        model_type = model.config.model_type

        from pos_shift import codeshell_pos_shift_attention_forward

        ATTENTION_FORWARD_MAPPING = {
            "codeshell": codeshell_pos_shift_attention_forward,
        }

        # Not all models require updated attention forwards
        if ATTENTION_FORWARD_MAPPING[model_type] is None:
            return

        def overwrite_forward(module) -> None:
            module.forward = types.MethodType(ATTENTION_FORWARD_MAPPING[model_type], module)

        return cls._call_modules_by_name(model, ATTENTION_NAME_MAPPING[model_type], overwrite_forward)

    @classmethod
    def _inject_attention_sink_kv_cache(cls, model, **attention_sink_kwargs) -> int:
        model_type = model.config.model_type
        attention_sink_kwargs["k_seq_dim"], attention_sink_kwargs["v_seq_dim"] = KV_DIM_MAPPING[model_type]

        def overwrite_forward(module):
            # Create the new cache
            module.attention_sink_kv_cache = AttentionSinkKVCache(**attention_sink_kwargs)

            # Keep track of the old forward method, we need it in the wrapped one
            old_forward = module.forward

            # Wrap the forward by overriding the past_key_values using the cache
            def wrapped_forward(self, *args, **kwargs):
                outputs = old_forward(*args, **kwargs)
                outputs.past_key_values = self.attention_sink_kv_cache(outputs.past_key_values)
                return outputs

            module.forward = types.MethodType(wrapped_forward, module)

        return cls._call_modules_by_name(model, MODEL_NAME_MAPPING[model_type], overwrite_forward)

    @classmethod
    def _call_modules_by_name(cls, module, target_name: str, func: Callable) -> int:
        if module.__class__.__name__ == target_name:
            func(module)
            return 1

        return sum(cls._call_modules_by_name(module, target_name, func) for module in module.children())
