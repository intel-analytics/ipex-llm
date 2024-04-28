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
from vllm.engine.ray_utils import initialize_ray_cluster
from vllm.entrypoints.llm import LLM
from vllm.utils import Counter

from ipex_llm.vllm.model_convert import _ipex_llm_convert
from ipex_llm.utils.common import invalidInputError


class IPEXLLMAsyncLLMEngine(AsyncLLMEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        load_in_low_bit: str = "sym_int4",
        # ipex_llm_optimize_mode: str = 'NATIVE',
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Enable ipex-llm optimizations
        _ipex_llm_convert(load_in_low_bit)
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        if parallel_config.worker_use_ray or engine_args.engine_use_ray:
            initialize_ray_cluster(parallel_config)
            # from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync
            from ipex_llm.vllm.ipex_llm_gpu_executor import get_gpu_executor_class_async
            executor_class = get_gpu_executor_class_async(load_in_low_bit)
        else:
            invalidInputError(parallel_config.world_size == 1, (
                "Ray is required if parallel_config.world_size > 1."))
            from vllm.executor.gpu_executor import GPUExecutorAsync
            executor_class = GPUExecutorAsync
        # Create the async LLM engine.
        engine = cls(parallel_config.worker_use_ray,
                     engine_args.engine_use_ray,
                     *engine_configs,
                     executor_class,
                     log_requests=not engine_args.disable_log_requests,
                     log_stats=not engine_args.disable_log_stats,
                     max_log_len=engine_args.max_log_len,
                     start_engine_loop=start_engine_loop)
        return engine


class IPEXLLMClass(LLM):
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
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
        max_context_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        load_in_low_bit: str = "sym_int4",
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
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
        load_in_low_bit: str = "sym_int4",
        # ipex_llm_optimize_mode: str = 'NATIVE',
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        _ipex_llm_convert(load_in_low_bit)
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]

        # Initialize the cluster and specify the executor class.
        if parallel_config.worker_use_ray:
            initialize_ray_cluster(parallel_config)
            # from vllm.executor.ray_gpu_executor import RayGPUExecutor
            from ipex_llm.vllm.ipex_llm_gpu_executor import get_gpu_executor_class
            executor_class = get_gpu_executor_class(load_in_low_bit)
        else:
            invalidInputError(parallel_config.world_size == 1,
                              "Ray is required if parallel_config.world_size > 1.")
            from vllm.executor.gpu_executor import GPUExecutor
            executor_class = GPUExecutor

        # Create the LLM engine.
        engine = cls(*engine_configs,
                     executor_class=executor_class,
                     log_stats=not engine_args.disable_log_stats)
        return engine
