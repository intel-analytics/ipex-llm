# Self-Speculative Decoding

### Speculative Decoding in Practice
In [speculative](https://arxiv.org/abs/2302.01318) [decoding](https://arxiv.org/abs/2211.17192), a small (draft) model quickly generates multiple draft tokens, which are then verified in parallel by the large (target) model. While speculative decoding can effectively speed up the target model, ***in practice it is difficult to maintain or even obtain a proper draft model***, especially when the target model is finetuned with customized data.

### Self-Speculative Decoding
Built on top of the concept of “[self-speculative decoding](https://arxiv.org/abs/2309.08168)”, IPEX-LLM can now accelerate the original FP16 or BF16 model ***without the need of a separate draft model or model finetuning***; instead, it automatically converts the original model to INT4, and uses the INT4 model as the draft model behind the scene. In practice, this brings ***~30% speedup*** for FP16 and BF16 LLM inference latency on Intel GPU and CPU respectively.

### Using IPEX-LLM Self-Speculative Decoding
Please refer to IPEX-LLM self-speculative decoding code snippets below, and the detailed [GPU](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/Speculative-Decoding) and [CPU](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Speculative-Decoding) examples in the project repo.

```python 
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             optimize_model=True,
                                             torch_dtype=torch.float16, #use bfloat16 on cpu
                                             load_in_low_bit="fp16", #use bf16 on cpu
                                             speculative=True, #set speculative to true
                                             trust_remote_code=True,
                                             use_cache=True)
output = model.generate(input_ids,
                        max_new_tokens=args.n_predict,
                        do_sample=False)
```