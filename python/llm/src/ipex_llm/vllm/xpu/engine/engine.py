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

from typing import TYPE_CHECKING, Any, ClassVar, Dict
from typing import List, Optional, Union
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.executor.ray_utils import initialize_ray_cluster
from vllm.entrypoints.llm import LLM
from vllm.utils import Counter
from ipex_llm.vllm.xpu.model_convert import _ipex_llm_convert
from ipex_llm.utils.common import invalidInputError
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from vllm.engine.metrics import (LoggingStatLogger, PrometheusStatLogger,
                                 StatLoggerBase, Stats)
from vllm.executor.executor_base import ExecutorBase
import vllm.envs as envs

# TODO: handle this next
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
        engine_configs = engine_args.create_engine_configs()

        _ipex_llm_convert(load_in_low_bit)
        parallel_config = engine_configs[2]
        if parallel_config.worker_use_ray or engine_args.engine_use_ray:
            initialize_ray_cluster(parallel_config)
            # from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync
            from ipex_llm.vllm.xpu.ipex_llm_gpu_executor import get_gpu_executor_class_async
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
            raise TypeError(
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
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        load_in_low_bit: str = "sym_int4",
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        _ipex_llm_convert(load_in_low_bit)
        # TODO: enable use of executor_class
        # TODO: replace this later
        executor_class = None
        distributed_executor_backend = (
            engine_config.parallel_config.distributed_executor_backend)
        # Initialize the cluster and specify the executor class.
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, ExecutorBase):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"ExecutorBase. Got {distributed_executor_backend}.")
            if distributed_executor_backend.uses_ray:  # type: ignore
                initialize_ray_cluster(engine_config.parallel_config)
            executor_class = distributed_executor_backend
        elif engine_config.device_config.device_type == "neuron":
            from vllm.executor.neuron_executor import NeuronExecutor
            executor_class = NeuronExecutor
        elif engine_config.device_config.device_type == "tpu":
            from vllm.executor.tpu_executor import TPUExecutor
            executor_class = TPUExecutor
        elif engine_config.device_config.device_type == "cpu":
            from vllm.executor.cpu_executor import CPUExecutor
            executor_class = CPUExecutor
        elif engine_config.device_config.device_type == "openvino":
            from vllm.executor.openvino_executor import OpenVINOExecutor
            executor_class = OpenVINOExecutor
        elif engine_config.device_config.device_type == "xpu":
            # TODO: enable these threes
            if distributed_executor_backend == "ray":
                initialize_ray_cluster(engine_config.parallel_config)
                from vllm.executor.ray_xpu_executor import RayXPUExecutor
                executor_class = RayXPUExecutor
            elif distributed_executor_backend == "mp":
                from vllm.executor.multiproc_xpu_executor import (
                    MultiprocessingXPUExecutor)
                executor_class = MultiprocessingXPUExecutor
            else:
                # Let's test this first
                from vllm.executor.xpu_executor import XPUExecutor
                executor_class = XPUExecutor
        elif distributed_executor_backend == "ray":
            initialize_ray_cluster(engine_config.parallel_config)
            from vllm.executor.ray_gpu_executor import RayGPUExecutor
            executor_class = RayGPUExecutor
        elif distributed_executor_backend == "mp":
            from vllm.executor.multiproc_gpu_executor import (
                MultiprocessingGPUExecutor)
            assert not envs.VLLM_USE_RAY_SPMD_WORKER, (
                "multiprocessing distributed executor backend does not "
                "support VLLM_USE_RAY_SPMD_WORKER=1")
            executor_class = MultiprocessingGPUExecutor
        else:
            from vllm.executor.gpu_executor import GPUExecutor
            executor_class = GPUExecutor
        ##################################END######################################
        # executor_class = cls._get_executor_cls(engine_config)
        # Create the LLM engine.
        engine = cls(
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

        return engine
