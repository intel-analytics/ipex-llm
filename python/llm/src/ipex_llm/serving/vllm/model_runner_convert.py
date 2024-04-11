from vllm.model_executor.model_loader import get_model
from .model_convert import _model_mlp_convert, _model_attention_convert


def _ipex_llm_load_model(self) -> None:
    _model_mlp_convert()
    _model_attention_convert()

    self.model = get_model(self.model_config,
                            self.device_config,
                            lora_config=self.lora_config,
                            parallel_config=self.parallel_config,
                            scheduler_config=self.scheduler_config)
    
    from ipex_llm import optimize_model
    optimize_model(self.model, low_bit="sym_int4", torch_dtype=self.model_config.dtype)


def _model_runner_convert(model_runner):
    setattr(model_runner, "load_model", _ipex_llm_load_model)