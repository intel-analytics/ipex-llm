.. meta::
   :google-site-verification: S66K6GAclKw1RroxU0Rka_2d1LZFVe27M0gRneEsIVI

################################################
The BigDL Project
################################################

------

************************************************
BigDL-LLM: low-Bit LLM library
************************************************

.. raw:: html

   <p>
      <a href="https://github.com/intel-analytics/BigDL/tree/main/python/llm"><code><span>bigdl-llm</span></code></a> is a library for running <strong>LLM</strong> (large language model) on Intel <strong>XPU</strong> (from <em>Laptop</em> to <em>GPU</em> to <em>Cloud</em>) using <strong>INT4</strong> with very low latency <sup><a href="#footnote-perf" id="ref-perf">[1]</a></sup> (for any <strong>PyTorch</strong> model).
   </p>

.. note::

   It is built on top of the excellent work of `llama.cpp <https://github.com/ggerganov/llama.cpp>`_, `gptq <https://github.com/IST-DASLab/gptq>`_, `bitsandbytes <https://github.com/TimDettmers/bitsandbytes>`_, `qlora <https://github.com/artidoro/qlora>`_, etc.

============================================
Latest update
============================================
- ``bigdl-llm`` now supports Intel Arc and Flex GPU; see the the latest GPU examples `here <https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/gpu>`_.
- ``bigdl-llm`` tutorial is released `here <https://github.com/intel-analytics/bigdl-llm-tutorial>`_.
- Over 20 models have been verified on ``bigdl-llm``, including *LLaMA/LLaMA2, ChatGLM/ChatGLM2, MPT, Falcon, Dolly-v1/Dolly-v2, StarCoder, Whisper, QWen, Baichuan,* and more; see the complete list `here <https://github.com/intel-analytics/BigDL/tree/main/python/llm/README.md#verified-models>`_.


============================================
``bigdl-llm`` demos
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
``bigdl-llm`` quickstart
============================================

- `CPU <#cpu-quickstart>`_
- `GPU <#gpu-quickstart>`_

--------------------------------------------
CPU Quickstart
--------------------------------------------

You may install ``bigdl-llm`` on Intel CPU as follows as follows:

.. code-block:: console

   pip install --pre --upgrade bigdl-llm[all]

.. note::

   ``bigdl-llm`` has been tested on Python 3.9.

You can then apply INT4 optimizations to any Hugging Face *Transformers* models as follows.

.. code-block:: python

   #load Hugging Face Transformers model with INT4 optimizations
   from bigdl.llm.transformers import AutoModelForCausalLM
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

You may install ``bigdl-llm`` on Intel GPU as follows as follows:

.. code-block:: console

   # below command will install intel_extension_for_pytorch==2.0.110+xpu as default
   # you can install specific ipex/torch version for your need
   pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu

.. note::

   ``bigdl-llm`` has been tested on Python 3.9.

You can then apply INT4 optimizations to any Hugging Face *Transformers* models on Intel GPU as follows.

.. code-block:: python

   #load Hugging Face Transformers model with INT4 optimizations
   from bigdl.llm.transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_4bit=True)

   #run the optimized model on Intel GPU
   model = model.to('xpu')

   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   input_ids = tokenizer.encode(input_str, ...).to('xpu')
   output_ids = model.generate(input_ids, ...)
   output = tokenizer.batch_decode(output_ids.cpu())

**For more details, please refer to the bigdl-llm** `Document <doc/LLM/index.html>`_, `Readme <https://github.com/intel-analytics/BigDL/tree/main/python/llm>`_, `Tutorial <https://github.com/intel-analytics/bigdl-llm-tutorial>`_ and `API Doc <doc/PythonAPI/LLM/index.html>`_.

------

************************************************
Overview of the complete BigDL project
************************************************
`BigDL <https://github.com/intel-analytics/bigdl>`_ seamlessly scales your data analytics & AI applications from laptop to cloud, with the following libraries:

- `LLM <https://github.com/intel-analytics/BigDL/tree/main/python/llm>`_: Low-bit (INT3/INT4/INT5/INT8) large language model library for Intel CPU/GPU
- `Orca <doc/Orca/index.html>`_: Distributed Big Data & AI (TF & PyTorch) Pipeline on Spark and Ray
- `Nano <doc/Nano/index.html>`_: Transparent Acceleration of Tensorflow & PyTorch Programs on Intel CPU/GPU
- `DLlib <doc/DLlib/index.html>`_: "Equivalent of Spark MLlib" for Deep Learning
- `Chronos <doc/Chronos/index.html>`_: Scalable Time Series Analysis using AutoML
- `Friesian <doc/Friesian/index.html>`_: End-to-End Recommendation Systems
- `PPML <doc/PPML/index.html>`_: Secure Big Data and AI (with SGX Hardware Security)

------

************************************************
Choosing the right BigDL library
************************************************

.. graphviz::

    digraph BigDLDecisionTree {
        graph [pad=0.1 ranksep=0.3 tooltip=" "]
        node [color="#0171c3" shape=box fontname="Arial" fontsize=14 tooltip=" "]
        edge [tooltip=" "]
        
        Feature1 [label="Hardware Secured Big Data & AI?"]
        Feature2 [label="Python vs. Scala/Java?"]
        Feature3 [label="What type of application?"]
        Feature4 [label="Domain?"]
        
        LLM[href="https://github.com/intel-analytics/BigDL/blob/main/python/llm" target="_blank" target="_blank" style="rounded,filled" fontcolor="#ffffff" tooltip="Go to BigDL-LLM document"]
        Orca[href="../doc/Orca/index.html" target="_blank" target="_blank" style="rounded,filled" fontcolor="#ffffff" tooltip="Go to BigDL-Orca document"]
        Nano[href="../doc/Nano/index.html" target="_blank" target="_blank" style="rounded,filled" fontcolor="#ffffff" tooltip="Go to BigDL-Nano document"]
        DLlib1[label="DLlib" href="../doc/DLlib/index.html" target="_blank" style="rounded,filled" fontcolor="#ffffff" tooltip="Go to BigDL-DLlib document"]
        DLlib2[label="DLlib" href="../doc/DLlib/index.html" target="_blank" style="rounded,filled" fontcolor="#ffffff" tooltip="Go to BigDL-DLlib document"]
        Chronos[href="../doc/Chronos/index.html" target="_blank" style="rounded,filled" fontcolor="#ffffff" tooltip="Go to BigDL-Chronos document"]
        Friesian[href="../doc/Friesian/index.html" target="_blank" style="rounded,filled" fontcolor="#ffffff" tooltip="Go to BigDL-Friesian document"]
        PPML[href="../doc/PPML/index.html" target="_blank" style="rounded,filled" fontcolor="#ffffff" tooltip="Go to BigDL-PPML document"]
        
        ArrowLabel1[label="No" fontsize=12 width=0.1 height=0.1 style=filled color="#c9c9c9"]
        ArrowLabel2[label="Yes" fontsize=12 width=0.1 height=0.1 style=filled color="#c9c9c9"]
        ArrowLabel3[label="Python" fontsize=12 width=0.1 height=0.1 style=filled color="#c9c9c9"]
        ArrowLabel4[label="Scala/Java" fontsize=12 width=0.1 height=0.1 style=filled color="#c9c9c9"]
        ArrowLabel5[label="Large Language Model" fontsize=12 width=0.1 height=0.1 style=filled color="#c9c9c9"]
        ArrowLabel6[label="Big Data + \n AI (TF/PyTorch)" fontsize=12 width=0.1 height=0.1 style=filled color="#c9c9c9"]
        ArrowLabel7[label="Accelerate \n TensorFlow / PyTorch" fontsize=12 width=0.1 height=0.1 style=filled color="#c9c9c9"]
        ArrowLabel8[label="DL for Spark MLlib" fontsize=12 width=0.1 height=0.1 style=filled color="#c9c9c9"]
        ArrowLabel9[label="High Level App Framework" fontsize=12 width=0.1 height=0.1 style=filled color="#c9c9c9"]
        ArrowLabel10[label="Time Series" fontsize=12 width=0.1 height=0.1 style=filled color="#c9c9c9"]
        ArrowLabel11[label="Recommender System" fontsize=12 width=0.1 height=0.1 style=filled color="#c9c9c9"]
        
        Feature1 -> ArrowLabel1[dir=none]
        ArrowLabel1 -> Feature2
        Feature1 -> ArrowLabel2[dir=none]
        ArrowLabel2 -> PPML
        
        Feature2 -> ArrowLabel3[dir=none]
        ArrowLabel3 -> Feature3
        Feature2 -> ArrowLabel4[dir=none]
        ArrowLabel4 -> DLlib1
        
        Feature3 -> ArrowLabel5[dir=none]
        ArrowLabel5 -> LLM
        Feature3 -> ArrowLabel6[dir=none]
        ArrowLabel6 -> Orca
        Feature3 -> ArrowLabel7[dir=none]
        ArrowLabel7 -> Nano
        Feature3 -> ArrowLabel8[dir=none]
        ArrowLabel8 -> DLlib2
        Feature3 -> ArrowLabel9[dir=none]
        ArrowLabel9 -> Feature4
     
        Feature4 -> ArrowLabel10[dir=none]
        ArrowLabel10 -> Chronos
        Feature4 -> ArrowLabel11[dir=none]
        ArrowLabel11 -> Friesian
    }

------

.. raw:: html

    <div>
        <p>
            <sup><a href="#ref-perf" id="footnote-perf">[1]</a>
               Performance varies by use, configuration and other factors. <code><span>bigdl-llm</span></code> may not optimize to the same degree for non-Intel products. Learn more at <a href="https://www.Intel.com/PerformanceIndex">www.Intel.com/PerformanceIndex</a>.
            </sup>
        </p>
    </div>
