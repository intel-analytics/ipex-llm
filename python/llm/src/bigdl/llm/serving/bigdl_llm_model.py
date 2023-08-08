from fastchat.model.model_adapter import register_model_adapter, BaseModelAdapter
from transformers import AutoTokenizer

is_fastchat_llm_patched = False

class BigDLLLMAdapter(BaseModelAdapter):
    "Model adapter for bigdl-llm backend models"

    def match(self, model_path: str):
        return "bigdl" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, revision=revision
        )
        from bigdl.llm.transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

def patch_fastchat_models():
    global is_fastchat_llm_patched
    if not is_fastchat_llm_patched:
        register_model_adapter(BigDLLLMAdapter)


if __name__ == '__main__':
    patch_fastchat_models()
