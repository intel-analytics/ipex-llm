# IPEX-LLM Examples: CPU

Here, we provide some examples on how you could apply IPEX-LLM INT4 optimizations on popular open-source models in the community.

To run these examples, please first refer to [here](./install_cpu.html) for more information about how to install ``ipex-llm``, requirements and best practices for setting up your environment.

The following models have been verified on either servers or laptops with Intel CPUs.

## Example of PyTorch API

| Model      | Example of PyTorch API                                |
|------------|-------------------------------------------------------|
| LLaMA 2    | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/PyTorch-Models/Model/llama2)  |
| ChatGLM    | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/PyTorch-Models/Model/chatglm) |
| Mistral    | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/PyTorch-Models/Model/mistral) |
| Bark       | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/PyTorch-Models/Model/bark)    |
| BERT       | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/PyTorch-Models/Model/bert)    |
| Openai Whisper    | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/PyTorch-Models/Model/openai-whisper) |

```eval_rst
.. important::

   In addition to INT4 optimization, IPEX-LLM also provides other low bit optimizations (such as INT8, INT5, NF4, etc.). You may apply other low bit optimizations through PyTorch API as `example <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/PyTorch-Models/More-Data-Types>`_.
```


## Example of `transformers`-style API

| Model      | Example of `transformers`-style API                   |
|------------|-------------------------------------------------------|
| LLaMA *(such as Vicuna, Guanaco, Koala, Baize, WizardLM, etc.)* | [link1](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Native-Models), [link2](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/vicuna) |
| LLaMA 2    | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/PyTorch-Models/Model/llama2) | [link1](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Native-Models), [link2](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama2) |
| ChatGLM    | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/PyTorch-Models/Model/chatglm) | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm)   |
| ChatGLM2   | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm2)  |
| Mistral    | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/mistral)   |
| Falcon     | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/falcon)    |
| MPT        | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/mpt)       |
| Dolly-v1   | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/dolly_v1)  |
| Dolly-v2   | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/dolly_v2)  |
| Replit Code| [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/replit)    |
| RedPajama  | [link1](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Native-Models), [link2](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/redpajama) |
| Phoenix    | [link1](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Native-Models), [link2](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/phoenix)   |
| StarCoder  | [link1](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Native-Models), [link2](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/starcoder) |
| Baichuan   | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/baichuan)  |
| Baichuan2  | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/baichuan2) |
| InternLM   | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm)  |
| Qwen       | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen)      |
| Aquila     | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/aquila)    |
| MOSS       | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/moss)      |
| Whisper    | [link](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/whisper)   |

```eval_rst
.. important::

   In addition to INT4 optimization, IPEX-LLM also provides other low bit optimizations (such as INT8, INT5, NF4, etc.). You may apply other low bit optimizations through ``transformers``-style API as `example <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/More-Data-Types>`_.
```


```eval_rst
.. seealso::

   See the complete examples `here <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU>`_.
```

