.. meta::
   :google-site-verification: S66K6GAclKw1RroxU0Rka_2d1LZFVe27M0gRneEsIVI

################################################
The IPEX Project
################################################

------

************************************************
IPEX-LLM
************************************************

.. raw:: html

   <p>
      <a href="https://github.com/intel-analytics/ipex-llm/"><code><span>ipex-llm</span></code></a> is a library for running <strong>LLM</strong> (large language model) on Intel <strong>XPU</strong> (from <em>Laptop</em> to <em>GPU</em> to <em>Cloud</em>) using <strong>INT4/FP4/INT8/FP8</strong> with very low latency <sup><a href="#footnote-perf" id="ref-perf">[1]</a></sup> (for any <strong>PyTorch</strong> model).
   </p>

.. note::

   It is built on top of the excellent work of `llama.cpp <https://github.com/ggerganov/llama.cpp>`_, `gptq <https://github.com/IST-DASLab/gptq>`_, `bitsandbytes <https://github.com/TimDettmers/bitsandbytes>`_, `qlora <https://github.com/artidoro/qlora>`_, etc.

============================================
Latest update 🔥
============================================
- [2024/03] **LangChain** added support for ``ipex-llm``; see the details `here <https://python.langchain.com/docs/integrations/llms/ipex>`_.
- [2024/02] ``ipex-llm`` now supports directly loading model from `ModelScope <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/ModelScope-Models>`_ (`魔搭 <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/ModelScope-Models>`_).
- [2024/02] ``ipex-llm`` added inital **INT2** support (based on llama.cpp `IQ2 <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/GGUF-IQ2>`_ mechanism), which makes it possible to run large-size LLM (e.g., Mixtral-8x7B) on Intel GPU with 16GB VRAM.
- [2024/02] Users can now use ``ipex-llm`` through `Text-Generation-WebUI <https://github.com/intel-analytics/text-generation-webui>`_ GUI.
- [2024/02] ``ipex-llm`` now supports `Self-Speculative Decoding <doc/LLM/Inference/Self_Speculative_Decoding.html>`_, which in practice brings **~30% speedup** for FP16 and BF16 inference latency on Intel `GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/Speculative-Decoding>`_ and `CPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Speculative-Decoding>`_ respectively.
- [2024/02] ``ipex-llm`` now supports a comprehensive list of LLM finetuning on Intel GPU (including `LoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/LoRA>`_, `QLoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QLoRA>`_, `DPO <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/DPO>`_, `QA-LoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QA-LoRA>`_ and `ReLoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/ReLora>`_).
- [2024/01] Using ``ipex-llm`` `QLoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QLoRA>`_, we managed to finetune LLaMA2-7B in **21 minutes** and LLaMA2-70B in **3.14 hours** on 8 Intel Max 1550 GPU for `Standford-Alpaca <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QLoRA/alpaca-qlora>`_ (see the blog `here <https://www.intel.com/content/www/us/en/developer/articles/technical/finetuning-llms-on-intel-gpus-using-ipex-llm.html>`_).
- [2024/01] 🔔🔔🔔 **The default** ``ipex-llm`` **GPU Linux installation has switched from PyTorch 2.0 to PyTorch 2.1, which requires new oneAPI and GPU driver versions. (See the** `GPU installation guide <https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html>`_ **for more details.)**
- [2023/12] ``ipex-llm`` now supports `ReLoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/ReLora>`_ (see `"ReLoRA: High-Rank Training Through Low-Rank Updates" <https://arxiv.org/abs/2307.05695>`_).
- [2023/12] ``ipex-llm`` now supports `Mixtral-8x7B <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/mixtral>`_ on both Intel `GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/mixtral>`_ and `CPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/mixtral>`_.
- [2023/12] ``ipex-llm`` now supports `QA-LoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QA-LoRA>`_ (see `"QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models" <https://arxiv.org/abs/2309.14717>`_).
- [2023/12] ``ipex-llm`` now supports `FP8 and FP4 inference <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/More-Data-Types>`_ on Intel **GPU**.
- [2023/11] Initial support for directly loading `GGUF <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/GGUF>`_, `AWQ <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/AWQ>`_ and `GPTQ <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/GPTQ>`_ models in to ``ipex-llm`` is available.
- [2023/11] ``ipex-llm`` now supports `vLLM continuous batching <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/vLLM-Serving>`_ on both Intel `GPU  <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/vLLM-Serving>`_ and `CPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/vLLM-Serving>`_.
- [2023/10] ``ipex-llm`` now supports `QLoRA finetuning <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QLoRA>`_ on both Intel `GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QLoRA>`_ and `CPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/QLoRA-FineTuning>`_.
- [2023/10] ``ipex-llm`` now supports `FastChat serving <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/src/ipex-llm/llm/serving>`_ on on both Intel CPU and GPU.
- [2023/09] ``ipex-llm`` now supports `Intel GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU>`_ (including Arc, Flex and MAX)
- [2023/09] ``ipex-llm`` `tutorial <https://github.com/intel-analytics/ipex-llm-tutorial>`_ is released.
- Over 30 models have been verified on ``ipex-llm``, including *LLaMA/LLaMA2, ChatGLM2/ChatGLM3, Mistral, Falcon, MPT, LLaVA, WizardCoder, Dolly, Whisper, Baichuan/Baichuan2, InternLM, Skywork, QWen/Qwen-VL, Aquila, MOSS* and more; see the complete list `here <https://github.com/intel-analytics/ipex#verified-models>`_.

============================================
``ipex-llm`` demos
============================================

See the **optimized performance** of ``chatglm2-6b`` and ``llama-2-13b-chat`` models on 12th Gen Intel Core CPU and Intel Arc GPU below.

.. raw:: html
   
   <table width="100%">
      <tr>
         <td align="center" colspan="2">12th Gen Intel Core CPU</td>
         <td align="center" colspan="2">Intel Arc GPU</td>
      </tr>
      <tr>
         <td>
            <a href="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-6b.gif"><img src="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-6b.gif" ></a>
         </td>
         <td>
            <a href="https://llm-assets.readthedocs.io/en/latest/_images/llama-2-13b-chat.gif"><img src="https://llm-assets.readthedocs.io/en/latest/_images/llama-2-13b-chat.gif"></a>
         </td>
         <td>
            <a href="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-arc.gif"><img src="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-arc.gif"></a>
         </td>
         <td>
            <a href="https://llm-assets.readthedocs.io/en/latest/_images/llama2-13b-arc.gif"><img src="https://llm-assets.readthedocs.io/en/latest/_images/llama2-13b-arc.gif"></a>
         </td>
      </tr>
      <tr>
         <td align="center" width="25%"><code>chatglm2-6b</code></td>
         <td align="center" width="25%"><code>llama-2-13b-chat</code></td>
         <td align="center" width="25%"><code>chatglm2-6b</code></td>
         <td align="center" width="25%"><code>llama-2-13b-chat</code></td>
      </tr>
   </table>

============================================
``ipex-llm`` quickstart
============================================

- `Windows GPU installation <doc/LLM/Quickstart/install_windows_gpu.html>`_
- `Run IPEX-LLM in Text-Generation-WebUI <doc/LLM/Quickstart/webui_quickstart.html>`_
- `Run IPEX-LLM using Docker <https://github.com/intel-analytics/ipex-llm/tree/main/docker/llm>`_
- `CPU quickstart <#cpu-quickstart>`_
- `GPU quickstart <#gpu-quickstart>`_

--------------------------------------------
CPU Quickstart
--------------------------------------------

You may install ``ipex-llm`` on Intel CPU as follows as follows:

.. note::

   See the `CPU installation guide <doc/LLM/Overview/install_cpu.html>`_ for more details.

.. code-block:: console

   pip install --pre --upgrade ipex-llm[all]

.. note::

   ``ipex-llm`` has been tested on Python 3.9, 3.10 and 3.11

You can then apply INT4 optimizations to any Hugging Face *Transformers* models as follows.

.. code-block:: python

   #load Hugging Face Transformers model with INT4 optimizations
   from ipex_llm.transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_4bit=True)

   #run the optimized model on Intel CPU
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   input_ids = tokenizer.encode(input_str, ...)
   output_ids = model.generate(input_ids, ...)
   output = tokenizer.batch_decode(output_ids)

--------------------------------------------
GPU Quickstart
--------------------------------------------

You may install ``ipex-llm`` on Intel GPU as follows as follows:

.. note::

   See the `GPU installation guide <doc/LLM/Overview/install_gpu.html>`_ for more details.

.. code-block:: console

   # below command will install intel_extension_for_pytorch==2.1.10+xpu as default
   pip install --pre --upgrade ipex-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu

.. note::

   ``ipex-llm`` has been tested on Python 3.9, 3.10 and 3.11

You can then apply INT4 optimizations to any Hugging Face *Transformers* models on Intel GPU as follows.

.. code-block:: python

   #load Hugging Face Transformers model with INT4 optimizations
   from ipex_llm.transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_4bit=True)

   #run the optimized model on Intel GPU
   model = model.to('xpu')

   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   input_ids = tokenizer.encode(input_str, ...).to('xpu')
   output_ids = model.generate(input_ids, ...)
   output = tokenizer.batch_decode(output_ids.cpu())

**For more details, please refer to the ipex-llm** `Document <doc/LLM/index.html>`_, `Readme <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm>`_, `Tutorial <https://github.com/intel-analytics/ipex-llm-tutorial>`_ and `API Doc <doc/PythonAPI/LLM/index.html>`_.