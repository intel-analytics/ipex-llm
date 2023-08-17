from fastchat.model.model_adapter import register_model_adapter, BaseModelAdapter, ChatGLMAdapter
from transformers import AutoTokenizer

is_fastchat_patched = False
_mapping_fastchat = None

def _get_patch_map():
    global _mapping_fastchat

    if _mapping_fastchat is None:
        _mapping_fastchat = []

    _mapping_fastchat += [
        [BaseModelAdapter, "load_model", load_model_base, None],
        [ChatGLMAdapter, "load_model", load_model_chatglm, None],
    ]

    return _mapping_fastchat


# TODO: Decide if we want to change the load_model method for base?
def load_model_base(self, model_path: str, from_pretrained_kwargs: dict):
    revision = from_pretrained_kwargs.get("revision", "main")
    # TODO: remove
    print("Customized bigdl-llm loader")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=self.use_fast_tokenizer,
        revision=revision,
    )
    from bigdl.llm.transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
    )
    return model, tokenizer

def load_model_chatglm(self, model_path: str, from_pretrained_kwargs: dict):
    revision = from_pretrained_kwargs.get("revision", "main")
    # TODO: remove
    print("Customized bigdl-llm loader")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, revision=revision
    )
    from bigdl.llm.transformers import AutoModel
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, load_in_4bit=True, **from_pretrained_kwargs
    )
    return model, tokenizer

class BigDLLLMAdapter(BaseModelAdapter):
    "Model adapter for bigdl-llm backend models"

    def match(self, model_path: str):
        return "bigdl" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, revision=revision
        )
        # TODO: remove
        print("Customized bigdl-llm loader")
        from bigdl.llm.transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

def patch_fastchat():
    global is_fastchat_patched
    if is_fastchat_patched:
        return
    register_model_adapter(BigDLLLMAdapter)
    mapping_fastchat = _get_patch_map()

    for mapping_iter in mapping_fastchat:
        if mapping_iter[3] is None:
            mapping_iter[3] = getattr(mapping_iter[0], mapping_iter[1], None)
        setattr(mapping_iter[0], mapping_iter[1], mapping_iter[2])

    is_fastchat_patched = True

# TODO: remove this later
if __name__ == '__main__':
    patch_fastchat()
    from fastchat.model.model_adapter import load_model
    # Let's try load chatglm model and other models to see if it worker correctly
    model, tokenizer = load_model("test", "cpu", 0)
