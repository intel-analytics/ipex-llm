from vllm.logger import init_logger
from vllm.executor.ray_utils import RayWorkerWrapper


logger = init_logger(__name__)


class IPEXLLMWrapper(RayWorkerWrapper):
    def __init__(self, load_in_low_bit="sym_int4", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from ipex_llm.vllm.cpu.model_convert import _ipex_llm_convert
        _ipex_llm_convert(load_in_low_bit=load_in_low_bit)
        self.compiled_dag_cuda_device_set = False


def get_ipex_llm_wrapper(load_in_low_bit):
    # The reason why we not using functools.partial is that
    # ray seems not work well with it.
    class WrapperWithLoadBit(IPEXLLMWrapper):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(load_in_low_bit=load_in_low_bit, *args, **kwargs)

    # a = functools.partial(IPEXLLMWrapper, load_in_low_bit=load_in_low_bit)
    return WrapperWithLoadBit
