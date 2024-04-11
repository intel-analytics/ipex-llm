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

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.ray_utils import initialize_ray_cluster
from vllm.usage.usage_lib import UsageContext

from .ipex_llm_convert import _ipex_llm_convert


class IPEXLLMAsyncLLMEngine(AsyncLLMEngine):
    def __init__(self, *args, **kwargs):
        from vllm.worker.cpu_worker import CPUModelRunner
        _ipex_llm_convert(CPUModelRunner)

        super().__init__(*args, **kwargs)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()

        if engine_config.device_config.device_type == "neuron":
            raise NotImplementedError("Neuron is not supported for "  # noqa
                                      "async engine yet.")
        elif engine_config.device_config.device_type == "cpu":
            from .cpu_executor import CPUExecutorAsync
            executor_class = CPUExecutorAsync
        elif (engine_config.parallel_config.worker_use_ray
              or engine_args.engine_use_ray):
            initialize_ray_cluster(engine_config.parallel_config)
            from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync
            executor_class = RayGPUExecutorAsync
        else:
            assert engine_config.parallel_config.world_size == 1, (  # noqa
                "Ray is required if parallel_config.world_size > 1.")
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


class IPEXLLMLLMEngine(LLMEngine):
    def __init__(self, *args, **kwargs):
        from vllm.worker.cpu_worker import CPUModelRunner
        _ipex_llm_convert(CPUModelRunner)

        super().__init__(*args, **kwargs)
