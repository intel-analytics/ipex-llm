pip install librosa
pip install git+https://github.com/huggingface/transformers # transformers              4.45.0.dev0

```
(qiyue-llm-0815) arda@arda-arc16:~/qiyue/my/ipex-llm/python/llm/example/GPU/HuggingFace/Multimodal/qwen2-audio$ python generate.py 
/home/arda/miniforge3/envs/qiyue-llm-0815/lib/python3.11/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: ''If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
  warn(
/home/arda/miniforge3/envs/qiyue-llm-0815/lib/python3.11/site-packages/transformers/deepspeed.py:24: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
2024-08-15 23:54:54,304 - INFO - intel_extension_for_pytorch auto imported
Loading checkpoint shards: 100%|█████████████████████████████| 5/5 [00:26<00:00,  5.31s/it]
2024-08-15 23:55:21,296 - INFO - Converting the current model to sym_int4 format......
It is strongly recommended to pass the `sampling_rate` argument to this function. Failing to do so can result in silent errors that might be hard to debug.
We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
1.9167637825012207
1.0979325771331787
1.0972487926483154
每个人都希望被赏识，所以如果你欣赏某人，不要保密。
```