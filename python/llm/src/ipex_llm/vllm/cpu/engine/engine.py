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

from typing import List, Optional, Union
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.llm import LLM
from vllm.executor.ray_utils import initialize_ray_cluster
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from vllm.utils import Counter

from ipex_llm.utils.common import invalidInputError


class IPEXLLMAsyncLLMEngine(AsyncLLMEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        load_in_low_bit: Optional[str] = None,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Enable ipex-llm optimizations
        engine_config = engine_args.create_engine_config()
        from ipex_llm.vllm.cpu.model_convert import _ipex_llm_convert
        _ipex_llm_convert(load_in_low_bit)
        if engine_config.device_config.device_type == "neuron":
            from vllm.executor.neuron_executor import NeuronExecutorAsync
            executor_class = NeuronExecutorAsync
        elif engine_config.device_config.device_type == "cpu":
            invalidInputError(not engine_config.parallel_config.worker_use_ray, (
                "Ray is not supported with the CPU backend."))
            from vllm.executor.cpu_executor import CPUExecutorAsync
            executor_class = CPUExecutorAsync
        elif engine_config.parallel_config.worker_use_ray:
            initialize_ray_cluster(engine_config.parallel_config)
            from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync
            executor_class = RayGPUExecutorAsync
        else:
            invalidInputError(engine_config.parallel_config.world_size == 1, (
                "Ray is required if parallel_config.world_size > 1."))
            from vllm.executor.gpu_executor import GPUExecutorAsync
            executor_class = GPUExecutorAsync
        # Create the async LLM engine.
        engine = cls(
            engine_config.parallel_config.worker_use_ray,
            engine_args.engine_use_ray,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            max_log_len=engine_args.max_log_len,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
        )
        return engine


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
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        load_in_low_bit: Optional[str] = None,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
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
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.llm_engine = IPEXLLMLLMEngine.from_engine_args(engine_args,
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
        load_in_low_bit: Optional[str] = None,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        from ipex_llm.vllm.cpu.model_convert import _ipex_llm_convert
        _ipex_llm_convert(load_in_low_bit)

        # Initialize the cluster and specify the executor class.
        if engine_config.device_config.device_type == "neuron":
            from vllm.executor.neuron_executor import NeuronExecutor
            executor_class = NeuronExecutor
        elif engine_config.device_config.device_type == "cpu":
            from vllm.executor.cpu_executor import CPUExecutor
            executor_class = CPUExecutor
        elif engine_config.parallel_config.worker_use_ray:
            initialize_ray_cluster(engine_config.parallel_config)
            from vllm.executor.ray_gpu_executor import RayGPUExecutor
            executor_class = RayGPUExecutor
        else:
            invalidInputError(engine_config.parallel_config.world_size == 1, (
                "Ray is required if parallel_config.world_size > 1."))
            from vllm.executor.gpu_executor import GPUExecutor
            executor_class = GPUExecutor

        # Create the LLM engine.
        engine = cls(**engine_config.to_dict(),
                     executor_class=executor_class,
                     log_stats=not engine_args.disable_log_stats,
                     usage_context=usage_context,
                     )
        return engine
