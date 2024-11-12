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
from typing import Dict, Optional
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.llm import LLM
from vllm.utils import Counter
from vllm.config import EngineConfig
from ipex_llm.vllm.xpu.model_convert import _ipex_llm_convert
from vllm.usage.usage_lib import UsageContext
from vllm.engine.metrics import StatLoggerBase
from vllm.engine.multiprocessing.engine import MQLLMEngine
import signal


class IPEXLLMAsyncLLMEngine(AsyncLLMEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        engine_config: Optional[EngineConfig] = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        load_in_low_bit: str = "sym_int4",
        stat_loggers: Optional[Dict[str, StatLoggerBase]]=None,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        _ipex_llm_convert(load_in_low_bit)
        return super().from_engine_args(engine_args=engine_args, engine_config=engine_config,
                                        start_engine_loop=start_engine_loop,
                                        usage_context=usage_context, stat_loggers=stat_loggers)


class IPEXLLMClass(LLM):
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        load_in_low_bit: str = "sym_int4",
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        removed_vision_keys = ("image_token_id", "image_feature_size",
                               "image_input_shape", "image_input_type")
        if any(k in kwargs for k in removed_vision_keys):
            raise TypeError(  # noqa
                "There is no need to pass vision-related arguments anymore.")
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.llm_engine = IPEXLLMLLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS,
            load_in_low_bit=load_in_low_bit)
        self.request_counter = Counter()


class IPEXLLMLLMEngine(LLMEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]]=None,
        load_in_low_bit: str = "sym_int4",
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        _ipex_llm_convert(load_in_low_bit)
        return super().from_engine_args(engine_args, usage_context, stat_loggers)


class IPEXLLMMQLLMEngine(MQLLMEngine):
    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs,
                         usage_context: UsageContext, ipc_path: str, load_in_low_bit: str):
        _ipex_llm_convert(load_in_low_bit)
        return super().from_engine_args(engine_args, usage_context, ipc_path)


def run_mp_engine(engine_args: AsyncEngineArgs, usage_context: UsageContext,
                  ipc_path: str, load_in_low_bit: str):

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm
        raise KeyboardInterrupt("MQLLMEngine terminated")  # noqa

    signal.signal(signal.SIGTERM, signal_handler)

    engine = IPEXLLMMQLLMEngine.from_engine_args(engine_args=engine_args,
                                                 usage_context=usage_context,
                                                 ipc_path=ipc_path,
                                                 load_in_low_bit=load_in_low_bit)
    engine.start()
