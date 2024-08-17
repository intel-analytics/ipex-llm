
import torch

def convert_forward(m, target_m, new_forward):
    if m.__class__ == target_m:
        bound_method = new_forward.__get__(m, m.__class__)
        setattr(m, "forward", bound_method)
    for _, sub_m in m.named_children():
        convert_forward(sub_m, target_m, new_forward)


def optimize_llm(model: torch.nn.Module, max_seq_len=1024, inter_pp=2, intra_pp=2, transpose_value_cache=True):
    if model.config.model_type == "llama":
        from ipex_llm.transformers.npu_models.llama_mp import gen_llama_fused_model_forward, DecodeRunner, PrefillRunner
        from transformers.models.llama.modeling_llama import LlamaModel
        decode_runner = DecodeRunner(model, max_seq_len, inter_pp=inter_pp, intra_pp=intra_pp, transpose_value_cache=transpose_value_cache)
        prefill_runner = PrefillRunner(model, max_seq_len, transpose_value_cache=transpose_value_cache)
        llama_model_forward = gen_llama_fused_model_forward(prefill_runner=prefill_runner,
                                                            decode_runner=decode_runner)
        convert_forward(model, LlamaModel, llama_model_forward)
