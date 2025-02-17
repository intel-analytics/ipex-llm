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
from vllm.logger import init_logger
from typing import Dict, Optional, Any, Union, Type
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.llm import LLM
from vllm.utils import Counter
from vllm.config import VllmConfig
from ipex_llm.vllm.xpu.model_convert import _ipex_llm_convert
from vllm.usage.usage_lib import UsageContext
from vllm.engine.metrics import StatLoggerBase
from vllm.engine.multiprocessing.engine import MQLLMEngine
import signal
from vllm.engine.arg_utils import (EngineArgs, HfOverrides, PoolerConfig,
                                   TaskOption)
from vllm.config import CompilationConfig
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
from vllm import envs
from vllm.v1.engine.async_llm import AsyncLLM
import os

logger = init_logger(__name__)


class IPEXLLMAsyncLLMEngine(AsyncLLMEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        engine_config: Optional[VllmConfig] = None,
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


class IPEXLLMAsyncV1Engine(AsyncLLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        engine_config: Optional[VllmConfig] = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        load_in_low_bit: str = "sym_int4",
        stat_loggers: Optional[Dict[str, StatLoggerBase]]=None,  # noqa
    ) -> "AsyncLLM":
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
        allowed_local_media_path: str = "",
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = True,
        hf_overrides: Optional[HfOverrides] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]]=None,
        # After positional args are removed, move this right below `model`
        task: TaskOption = "auto",
        override_pooler_config: Optional[PoolerConfig] = None,
        compilation_config: Optional[Union[int, Dict[str, Any]]]=None,
        load_in_low_bit: str = "sym_int4",
        **kwargs,
    ) -> None:
        '''
        LLM constructor.

        Note: if enforce_eager is unset (enforce_eager is None)
        it defaults to False.
        '''
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if compilation_config is not None:
            if isinstance(compilation_config, (int, dict)):
                compilation_config_instance = CompilationConfig.from_cli(
                    str(compilation_config))
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = None

        engine_args = EngineArgs(
            model=model,
            task=task,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
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
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            override_pooler_config=override_pooler_config,
            compilation_config=compilation_config_instance,
            **kwargs,
        )
        # Logic to switch between engines is done at runtime instead of import
        # to avoid import order issues
        self.engine_class = self.get_engine_class()
        self.llm_engine = self.engine_class.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS,
            load_in_low_bit=load_in_low_bit)

        self.request_counter = Counter()

    @staticmethod
    def get_engine_class() -> Type[LLMEngine]:
        if envs.VLLM_USE_V1:
            return IPEXLLMLLMV1Engine
        return IPEXLLMLLMEngine


class IPEXLLMLLMV1Engine(V1LLMEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]]=None,
        enable_multiprocessing: bool = False,
        load_in_low_bit: str = "sym_int4",
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.

        _ipex_llm_convert(load_in_low_bit)
        return super().from_engine_args(engine_args,
                                        usage_context,
                                        stat_loggers,
                                        enable_multiprocessing)


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
                  ipc_path: str, load_in_low_bit: str, engine_alive):

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm
        raise KeyboardInterrupt("MQLLMEngine terminated")  # noqa

    try:
        signal.signal(signal.SIGTERM, signal_handler)

        engine = IPEXLLMMQLLMEngine.from_engine_args(engine_args=engine_args,
                                                     usage_context=usage_context,
                                                     ipc_path=ipc_path,
                                                     load_in_low_bit=load_in_low_bit)
        engine.start()
    except BaseException as e:
        logger.exception(e)
        engine_alive.value = False
        raise e  # noqa

if os.getenv("VLLM_USE_V1"):
    IPEXLLMAsyncLLMEngine = IPEXLLMAsyncV1Engine
