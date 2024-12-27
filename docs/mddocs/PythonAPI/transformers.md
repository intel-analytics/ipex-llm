# IPEX-LLM `transformers`-style API

## Hugging Face `transformers` AutoModel

You can apply IPEX-LLM optimizations on any Hugging Face Transformers models by using the standard AutoModel APIs.

> [!NOTE]
> Here we take `ipex_llm.transformers.AutoModelForCausalLM` as an example. The class method for the following class, including `ipex_llm.transformers.AutoModel` / `AutoModelForSpeechSeq2Seq` / `AutoModelForSeq2SeqLM` / `AutoModelForSequenceClassification` / `AutoModelForMaskedLM` / `AutoModelForQuestionAnswering` / `AutoModelForNextSentencePrediction` / `AutoModelForMultipleChoice` / `AutoModelForTokenClassification`, are the same.

### _`class`_ **`ipex_llm.transformers.AutoModelForCausalLM`**

#### _`classmethod`_ **`from_pretrained`**_`(*args, **kwargs)`_

Load a model from a directory or the HF Hub. Use load_in_4bit or load_in_low_bit parameter the weight of model’s linears can be loaded to low-bit format, like int4, int5 and int8.

Three new arguments are added to extend Hugging Face’s from_pretrained method as follows:

- **Parameters**:

  - **load_in_4bit**: boolean value, True means loading linear's weight to symmetric int 4 if the model is a regular fp16/bf16/fp32 model, and to asymmetric int 4 if the model is GPTQ model. Default to be `False`.

  - **load_in_low_bit**: `str` value, options are `'sym_int4'`, `'asym_int4'`, `'sym_int5'`, `'asym_int5'`, `'sym_int8'`, `'nf3'`, `'nf4'`, `'fp4'`, `'fp8'`, `'fp8_e4m3'`, `'fp8_e5m2'`, `'fp6'`, `'gguf_iq2_xxs'`, `'gguf_iq2_xs'`, `'gguf_iq1_s'`, `'gguf_q4k_m'`, `'gguf_q4k_s'`, `'fp16'`, `'bf16'`, `'fp6_k'`, `'sym_int4'` means symmetric int 4, `'asym_int4'` means asymmetric int 4, `'nf4'` means 4-bit NormalFloat, etc. Relevant low bit optimizations will be applied to the model.

  - **optimize_model**: boolean value, Whether to further optimize the low_bit llm model. Default to be `True`.

  - **modules_to_not_convert**: list of str value, modules (`nn.Module`) that are skipped when conducting model optimizations. Default to be `None`.

  - **speculative**: `boolean` value, Whether to use speculative decoding. Default to be `False`.

  - **cpu_embedding**: Whether to replace the Embedding layer, may need to set it to `True` when running IPEX-LLM on GPU. Default to be `False`.

  - **imatrix**: `str` value, represent filename of importance matrix pretrained on specific datasets for use with the improved quantization methods recently added to llama.cpp.

  - **model_hub**: `str` value, options are `'huggingface'` and `'modelscope'`, specify the model hub. Default to be `'huggingface'`.

  - **embedding_qtype**: `str` value, options are `'q2_k'`, `'q4_k'` now. Default to be `None`. Relevant low bit optimizations will be applied to `nn.Embedding` layer.

  - **mixed_precision**: `boolean` value, Whether to use mixed precision quantization. Default to be `False`. If set to `True`, we will use `sym_int8` for lm_head when `load_in_low_bit` is `sym_int4` or `asym_int4`.

  - **pipeline_parallel_stages**: `int` value, the number of GPUs allocated for pipeline parallel. Default to be `1`. Please set `pipeline_parallel_stages > 1` to run pipeline parallel inference on multiple GPUs.

- **Returns**：A model instance

#### _`classmethod`_ **`from_gguf`**_`(fpath, optimize_model=True, cpu_embedding=False, low_bit="sym_int4")`_

Load gguf model and tokenizer and convert it to bigdl-llm model and huggingface tokenzier

- **Parameters**:

  - **fpath**: Path to gguf model file

  - **optimize_model**: Whether to further optimize llm model, defaults to `True`

  - **cpu_embedding**: Whether to replace the Embedding layer, may need to set it to `True` when running IPEX-LLM on GPU, defaults to `False`

- **Returns**：An optimized ipex-llm model and a huggingface tokenizer

#### _`classmethod`_ **`load_convert`**_`(q_k, optimize_model, *args, **kwargs)`_

#### _`classmethod`_ **`load_low_bit`**_`(pretrained_model_name_or_path, *model_args, **kwargs)`_

Load a low bit optimized model (including INT4, INT5 and INT8) from a saved ckpt.

- **Parameters**:

  - **pretrained_model_name_or_path**: `str` value, Path to load the optimized model ckpt.

  - **optimize_model**: `boolean` value, Whether to further optimize the low_bit llm model.
                          Default to be `True`.

- **Returns**：A model instance
