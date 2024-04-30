.. meta::
   :google-site-verification: S66K6GAclKw1RroxU0Rka_2d1LZFVe27M0gRneEsIVI

.. important::
   
   .. raw:: html

      <p>
         <strong><em>
            <code><span>bigdl-llm</span></code> has now become <code><span>ipex-llm</span></code> (see the migration guide <a href="doc/LLM/Quickstart/bigdl_llm_migration.html">here</a>); you may find the original <code><span>BigDL</span></code> project <a href="https://github.com/intel-analytics/BigDL-2.x">here</a>.
         </em></strong>
      </p>

------

################################################
💫 IPEX-LLM
################################################

.. raw:: html

   <p>
      <strong><code><span><a href="https://github.com/intel-analytics/ipex-llm">IPEX-LLM</a></span></code></strong> is a PyTorch library for running <strong>LLM</strong> on Intel CPU and GPU <em>(e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)</em> with very low latency <sup><a href="#footnote-perf" id="ref-perf">[1]</a></sup>.
   </p>

.. note::

   .. raw:: html

       <p>
         <ul>
            <li><em>
               It is built on top of <strong>Intel Extension for PyTorch</strong> (<strong><code><span>IPEX</span></code></strong>), as well as the excellent work of <strong><code><span>llama.cpp</span></code></strong>, <strong><code><span>bitsandbytes</span></code></strong>, <strong><code><span>vLLM</span></code></strong>, <strong><code><span>qlora</span></code></strong>, <strong><code><span>AutoGPTQ</span></code></strong>, <strong><code><span>AutoAWQ</span></code></strong>, etc. 
            </li></em>
            <li><em>
               It provides seamless integration with <a href=doc/LLM/Quickstart/llama_cpp_quickstart.html>llama.cpp</a>, <a href=doc/LLM/Quickstart/ollama_quickstart.html>ollama</a>, <a href=doc/LLM/Quickstart/webui_quickstart.html>Text-Generation-WebUI</a>, <a href=https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels>HuggingFace transformers</a>, <a href=https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning>HuggingFace PEFT</a>, <a href=https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LangChain >LangChain</a>, <a href=https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LlamaIndex >LlamaIndex</a>, <a href=https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/Deepspeed-AutoTP >DeepSpeed-AutoTP</a>, <a href=https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/vLLM-Serving >vLLM</a>, <a href=https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/src/ipex_llm/serving/fastchat>FastChat</a>, <a href=https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/DPO>HuggingFace TRL</a>, <a href=https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Applications/autogen >AutoGen</a>, <a href=https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/ModelScope-Models >ModeScope</a>, etc.
            </li></em>
            <li><em>
               <strong>50+ models</strong> have been optimized/verified on <code><span>ipex-llm</span></code> (including LLaMA2, Mistral, Mixtral, Gemma, LLaVA, Whisper, ChatGLM, Baichuan, Qwen, RWKV, and more); see the complete list <a href=#verified-models>here</a>.
            </li></em>
         </ul>
      </p>

************************************************
Latest update 🔥
************************************************
* [2024/04] You can now run **Llama 3** on Intel GPU using ``llama.cpp`` and ``ollama``; see the quickstart `here <doc/LLM/Quickstart/llama3_llamacpp_ollama_quickstart.html>`_.
* [2024/04] ``ipex-llm`` now supports **Llama 3** on Intel `GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/llama3>`_ and `CPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3>`_.
* [2024/04] ``ipex-llm`` now provides C++ interface, which can be used as an accelerated backend for running `llama.cpp <doc/LLM/Quickstart/llama_cpp_quickstart.html>`_ and `ollama <doc/LLM/Quickstart/ollama_quickstart.html>`_ on Intel GPU.
* [2024/03] ``bigdl-llm`` has now become ``ipex-llm`` (see the migration guide `here <doc/LLM/Quickstart/bigdl_llm_migration.html>`_); you may find the original ``BigDL`` project `here <https://github.com/intel-analytics/bigdl-2.x>`_.
* [2024/02] ``ipex-llm`` now supports directly loading model from `ModelScope <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/ModelScope-Models>`_ (`魔搭 <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/ModelScope-Models>`_).
* [2024/02] ``ipex-llm`` added inital **INT2** support (based on llama.cpp `IQ2 <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/GGUF-IQ2>`_ mechanism), which makes it possible to run large-size LLM (e.g., Mixtral-8x7B) on Intel GPU with 16GB VRAM.
* [2024/02] Users can now use ``ipex-llm`` through `Text-Generation-WebUI <https://github.com/intel-analytics/text-generation-webui>`_ GUI.
* [2024/02] ``ipex-llm`` now supports `Self-Speculative Decoding <doc/LLM/Inference/Self_Speculative_Decoding.html>`_, which in practice brings **~30% speedup** for FP16 and BF16 inference latency on Intel `GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/Speculative-Decoding>`_ and `CPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Speculative-Decoding>`_ respectively.
* [2024/02] ``ipex-llm`` now supports a comprehensive list of LLM finetuning on Intel GPU (including `LoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/LoRA>`_, `QLoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QLoRA>`_, `DPO <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/DPO>`_, `QA-LoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QA-LoRA>`_ and `ReLoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/ReLora>`_).
* [2024/01] Using ``ipex-llm`` `QLoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QLoRA>`_, we managed to finetune LLaMA2-7B in **21 minutes** and LLaMA2-70B in **3.14 hours** on 8 Intel Max 1550 GPU for `Standford-Alpaca <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QLoRA/alpaca-qlora>`_ (see the blog `here <https://www.intel.com/content/www/us/en/developer/articles/technical/finetuning-llms-on-intel-gpus-using-ipex-llm.html>`_).


.. dropdown:: More updates
   :color: primary

   * [2023/12] ``ipex-llm`` now supports `ReLoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/ReLora>`_ (see `"ReLoRA: High-Rank Training Through Low-Rank Updates" <https://arxiv.org/abs/2307.05695>`_).
   * [2023/12] ``ipex-llm`` now supports `Mixtral-8x7B <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/mixtral>`_ on both Intel `GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/mixtral>`_ and `CPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/mixtral>`_.
   * [2023/12] ``ipex-llm`` now supports `QA-LoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QA-LoRA>`_ (see `"QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models" <https://arxiv.org/abs/2309.14717>`_).
   * [2023/12] ``ipex-llm`` now supports `FP8 and FP4 inference <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/More-Data-Types>`_ on Intel **GPU**.
   * [2023/11] Initial support for directly loading `GGUF <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/GGUF>`_, `AWQ <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/AWQ>`_ and `GPTQ <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/GPTQ>`_ models in to ``ipex-llm`` is available.
   * [2023/11] ``ipex-llm`` now supports `vLLM continuous batching <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/vLLM-Serving>`_ on both Intel `GPU  <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/vLLM-Serving>`_ and `CPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/vLLM-Serving>`_.
   * [2023/10] ``ipex-llm`` now supports `QLoRA finetuning <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QLoRA>`_ on both Intel `GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QLoRA>`_ and `CPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/QLoRA-FineTuning>`_.
   * [2023/10] ``ipex-llm`` now supports `FastChat serving <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/src/ipex-llm/llm/serving>`_ on on both Intel CPU and GPU.
   * [2023/09] ``ipex-llm`` now supports `Intel GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU>`_ (including iGPU, Arc, Flex and MAX).
   * [2023/09] ``ipex-llm`` `tutorial <https://github.com/intel-analytics/ipex-llm-tutorial>`_ is released.

************************************************
``ipex-llm`` Demos
************************************************

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

************************************************
``ipex-llm`` Quickstart
************************************************

============================================
Install ``ipex-llm``
============================================

* `Windows GPU <doc/LLM/Quickstart/install_windows_gpu.html>`_: installing ``ipex-llm`` on Windows with Intel GPU
* `Linux GPU <doc/LLM/Quickstart/install_linux_gpu.html>`_: installing ``ipex-llm`` on Linux with Intel GPU
* `Docker <https://github.com/intel-analytics/ipex-llm/tree/main/docker/llm>`_: using ``ipex-llm`` dockers on Intel CPU and GPU

.. seealso::

   For more details, please refer to the `installation guide <doc/LLM/Overview/install.html>`_

============================================
Run ``ipex-llm``
============================================

* `llama.cpp <doc/LLM/Quickstart/llama_cpp_quickstart.html>`_: running **llama.cpp** (*using C++ interface of* ``ipex-llm`` *as an accelerated backend for* ``llama.cpp``) on Intel GPU
* `ollama <doc/LLM/Quickstart/ollama_quickstart.html>`_: running **ollama** (*using C++ interface of* ``ipex-llm`` *as an accelerated backend for* ``ollama``) on Intel GPU
* `vLLM <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/vLLM-Serving>`_: running ``ipex-llm`` in ``vLLM`` on both Intel `GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/vLLM-Serving>`_ and `CPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/vLLM-Serving>`_
* `FastChat <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/src/ipex_llm/serving/fastchat>`_: running ``ipex-llm`` in ``FastChat`` serving on on both Intel GPU and CPU
* `LangChain-Chatchat RAG <https://github.com/intel-analytics/Langchain-Chatchat>`_: running ``ipex-llm`` in ``LangChain-Chatchat`` (*Knowledge Base QA using* **RAG** *pipeline*)
* `Text-Generation-WebUI <doc/LLM/Quickstart/webui_quickstart.html>`_: running ``ipex-llm`` in ``oobabooga`` **WebUI**
* `Benchmarking <doc/LLM/Quickstart/benchmark_quickstart.html>`_: running  (latency and throughput) benchmarks for ``ipex-llm`` on Intel CPU and GPU

============================================
Code Examples
============================================
* Low bit inference

  * `INT4 inference <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model>`_: **INT4** LLM inference on Intel `GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model>`_ and `CPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model>`_
  * `FP8/FP4 inference <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/More-Data-Types>`_: **FP8** and **FP4** LLM inference on Intel `GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/More-Data-Types>`_
  * `INT8 inference <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/More-Data-Types>`_: **INT8** LLM inference on Intel `GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/More-Data-Types>`_ and `CPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/More-Data-Types>`_
  * `INT2 inference <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/GGUF-IQ2>`_: **INT2** LLM inference (based on llama.cpp IQ2 mechanism) on Intel `GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/GGUF-IQ2>`_

* FP16/BF16 inference

  * **FP16** LLM inference on Intel `GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/Speculative-Decoding>`_, with possible `self-speculative decoding <doc/LLM/Inference/Self_Speculative_Decoding.html>`_ optimization
  * **BF16** LLM inference on Intel `CPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Speculative-Decoding>`_, with possible `self-speculative decoding <doc/LLM/Inference/Self_Speculative_Decoding.html>`_ optimization 

* Save and load

  * `Low-bit models <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Save-Load>`_: saving and loading ``ipex-llm`` low-bit models
  * `GGUF <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/GGUF>`_: directly loading GGUF models into ``ipex-llm``
  * `AWQ <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/AWQ>`_: directly loading AWQ models into ``ipex-llm``
  * `GPTQ <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/GPTQ>`_: directly loading GPTQ models into ``ipex-llm``

* Finetuning

  * LLM finetuning on Intel `GPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning>`_, including `LoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/LoRA>`_, `QLoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QLoRA>`_, `DPO <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/DPO>`_, `QA-LoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QA-LoRA>`_ and `ReLoRA <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/ReLora>`_
  * QLoRA finetuning on Intel `CPU <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/QLoRA-FineTuning>`_

* Integration with community libraries

  * `HuggingFace tansformers <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels>`_
  * `Standard PyTorch model <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/PyTorch-Models>`_
  * `DeepSpeed-AutoTP <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/Deepspeed-AutoTP>`_
  * `HuggingFace PEFT <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/HF-PEFT>`_
  * `HuggingFace TRL <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/DPO>`_
  * `LangChain <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LangChain>`_
  * `LlamaIndex <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LlamaIndex>`_
  * `AutoGen <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Applications/autogen>`_
  * `ModeScope <https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/ModelScope-Models>`_

* `Tutorials <https://github.com/intel-analytics/ipex-llm-tutorial>`_


.. seealso::

   For more details, please refer to the |ipex_llm_document|_.

.. |ipex_llm_document| replace:: ``ipex-llm`` document
.. _ipex_llm_document: doc/LLM/index.html


************************************************
Verified Models
************************************************

.. raw:: html

   <table class="table">
     <colgroup>
       <col style="width: 64%" />
       <col style="width: 18%" />
       <col style="width: 18%" /></colgroup>
     <thead>
       <tr>
         <th class="head">Model</th>
         <th class="head">CPU Example</th>
         <th class="head">GPU Example</th></tr>
     </thead>
     <tbody>
       <tr>
         <td>LLaMA
           <em>(such as Vicuna, Guanaco, Koala, Baize, WizardLM, etc.)</em></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Native-Models">link1</a>,
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/vicuna">link2</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/vicuna">link</a>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/vicuna">link</a></td>
       </tr>
       <tr>
         <td>LLaMA 2</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Native-Models">link1</a>,
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama2">link2</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/llama2">link</a>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/llama2">link</a></td>
       </tr>
       <tr>
         <td>LLaMA 3</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/llama3">link</a></td>
       </tr>
       <tr>
         <td>ChatGLM</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm">link</a></td>
         <td></td>
       </tr>
       <tr>
         <td>ChatGLM2</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm2">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/chatglm2">link</a></td>
       </tr>
       <tr>
         <td>ChatGLM3</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm3">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/chatglm3">link</a></td>
       </tr>
       <tr>
         <td>Mistral</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/mistral">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/mistral">link</a></td>
       </tr>
       <tr>
         <td>Mixtral</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/mixtral">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/mixtral">link</a></td>
       </tr>
       <tr>
         <td>Falcon</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/falcon">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/falcon">link</a></td>
       </tr>
       <tr>
         <td>MPT</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/mpt">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/mpt">link</a></td>
       </tr>
       <tr>
         <td>Dolly-v1</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/dolly_v1">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/dolly-v1">link</a></td>
       </tr>
       <tr>
         <td>Dolly-v2</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/dolly_v2">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/dolly-v2">link</a></td>
       </tr>
       <tr>
         <td>Replit Code</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/replit">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/replit">link</a></td>
       </tr>
       <tr>
         <td>RedPajama</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Native-Models">link1</a>,
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/redpajama">link2</a></td>
         <td></td>
       </tr>
       <tr>
         <td>Phoenix</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Native-Models">link1</a>,
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/phoenix">link2</a></td>
         <td></td>
       </tr>
       <tr>
         <td>StarCoder</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Native-Models">link1</a>,
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/starcoder">link2</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/starcoder">link</a></td>
       </tr>
       <tr>
         <td>Baichuan</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/baichuan">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/baichuan">link</a></td>
       </tr>
       <tr>
         <td>Baichuan2</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/baichuan2">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/baichuan2">link</a></td>
       </tr>
       <tr>
         <td>InternLM</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/internlm">link</a></td>
       </tr>
       <tr>
         <td>Qwen</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/qwen">link</a></td>
       </tr>
       <tr>
         <td>Qwen1.5</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen1.5">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/qwen1.5">link</a></td>
       </tr>
       <tr>
         <td>Qwen-VL</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen-vl">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/qwen-vl">link</a></td>
       </tr>
       <tr>
         <td>Aquila</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/aquila">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/aquila">link</a></td>
       </tr>
       <tr>
         <td>Aquila2</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/aquila2">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/aquila2">link</a></td>
       </tr>
       <tr>
         <td>MOSS</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/moss">link</a></td>
         <td></td>
       </tr>
       <tr>
         <td>Whisper</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/whisper">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/whisper">link</a></td>
       </tr>
       <tr>
         <td>Phi-1_5</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/phi-1_5">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/phi-1_5">link</a></td>
       </tr>
       <tr>
         <td>Flan-t5</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/flan-t5">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/flan-t5">link</a></td>
       </tr>
       <tr>
         <td>LLaVA</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/PyTorch-Models/Model/llava">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/PyTorch-Models/Model/llava">link</a></td>
       </tr>
       <tr>
         <td>CodeLlama</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/codellama">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/codellama">link</a></td>
       </tr>
       <tr>
         <td>Skywork</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/skywork">link</a></td>
         <td></td>
       </tr>
       <tr>
         <td>InternLM-XComposer</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm-xcomposer">link</a></td>
         <td></td>
       </tr>
       <tr>
         <td>WizardCoder-Python</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/wizardcoder-python">link</a></td>
         <td></td>
       </tr>
       <tr>
         <td>CodeShell</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/codeshell">link</a></td>
         <td></td>
       </tr>
       <tr>
         <td>Fuyu</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/fuyu">link</a></td>
         <td></td>
       </tr>
       <tr>
         <td>Distil-Whisper</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/distil-whisper">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/distil-whisper">link</a></td>
       </tr>
       <tr>
         <td>Yi</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/yi">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/yi">link</a></td>
       </tr>
       <tr>
         <td>BlueLM</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/bluelm">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/bluelm">link</a></td>
       </tr>
       <tr>
         <td>Mamba</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/PyTorch-Models/Model/mamba">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/PyTorch-Models/Model/mamba">link</a></td>
       </tr>
       <tr>
         <td>SOLAR</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/solar">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/solar">link</a></td>
       </tr>
       <tr>
         <td>Phixtral</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/phixtral">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/phixtral">link</a></td>
       </tr>
       <tr>
         <td>InternLM2</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm2">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/internlm2">link</a></td>
       </tr>
       <tr>
         <td>RWKV4</td>
         <td></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/rwkv4">link</a></td>
       </tr>
       <tr>
         <td>RWKV5</td>
         <td></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/rwkv5">link</a></td>
       </tr>
       <tr>
         <td>Bark</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/PyTorch-Models/Model/bark">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/PyTorch-Models/Model/bark">link</a></td>
       </tr>
       <tr>
         <td>SpeechT5</td>
         <td></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/PyTorch-Models/Model/speech-t5">link</a></td>
       </tr>
       <tr>
         <td>DeepSeek-MoE</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/deepseek-moe">link</a></td>
         <td></td>
       </tr>
       <tr>
         <td>Ziya-Coding-34B-v1.0</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/ziya">link</a></td>
         <td></td>
       </tr>
       <tr>
         <td>Phi-2</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/phi-2">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/phi-2">link</a></td>
       </tr>
       <tr>
         <td>Phi-3</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/phi-3">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/phi-3">link</a></td>
       </tr>
       <tr>
         <td>Yuan2</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/yuan2">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/yuan2">link</a></td>
       </tr>
       <tr>
         <td>Gemma</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/gemma">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/gemma">link</a></td>
       </tr>
       <tr>
         <td>DeciLM-7B</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/deciLM-7b">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/deciLM-7b">link</a></td>
       </tr>
       <tr>
         <td>Deepseek</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/deepseek">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/deepseek">link</a></td>
       </tr>
       <tr>
         <td>StableLM</td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/stablelm">link</a></td>
         <td>
           <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HF-Transformers-AutoModels/Model/stablelm">link</a></td>
       </tr>
     </tbody>
   </table>

************************************************
Get Support
************************************************
* Please report a bug or raise a feature request by opening a `Github Issue <https://github.com/intel-analytics/ipex-llm/issues>`_
* Please report a vulnerability by opening a draft `GitHub Security Advisory <https://github.com/intel-analytics/ipex-llm/security/advisories>`_

------

.. raw:: html

    <div>
        <p>
            <sup><a href="#ref-perf" id="footnote-perf">[1]</a>
               Performance varies by use, configuration and other factors. <code><span>ipex-llm</span></code> may not optimize to the same degree for non-Intel products. Learn more at <a href="https://www.Intel.com/PerformanceIndex">www.Intel.com/PerformanceIndex</a>.
            </sup>
        </p>
    </div>
