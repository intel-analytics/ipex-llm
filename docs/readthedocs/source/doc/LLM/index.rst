BigDL-LLM
=========================

.. raw:: html

    <p>
        <strong>BigDL-LLM</strong> is a library for running <strong><em>LLM</em></strong> (Large Language Models) on your Intel <strong><em>laptop</em></strong> using INT4 with very low latency <sup><a href="#footnote-perf" id="ref-perf">[1]</a></sup> (for any Hugging Face <em>Transformers</em> model).
    </p>

See the **optimized performance** of ``chatglm2-6b``, ``llama-2-13b-chat``, and ``starcoder-15b`` models on a 12th Gen Intel Core CPU below.

.. grid:: 1 3 3 3
    :gutter: 2

    .. grid-item::
        :class: sd-text-center

        .. image:: https://github.com/bigdl-project/bigdl-project.github.io/raw/master/assets/chatglm2-6b.gif
           :target: https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/chatglm2-6b.gif
        
        chatglm2-6b

    .. grid-item::
        :class: sd-text-center

        .. image:: https://github.com/bigdl-project/bigdl-project.github.io/raw/master/assets/llama-2-13b-chat.gif
           :target: https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/llama-2-13b-chat.gif
        
        llama-2-13b-chat

    .. grid-item::
        :class: sd-text-center

        .. image:: https://github.com/bigdl-project/bigdl-project.github.io/raw/master/assets/llm-15b5.gif
           :target: https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/llm-15b5.gif
        
        starcoder-15.5b

-------

.. grid:: 1 2 2 2
    :gutter: 2

    .. grid-item-card::

        **Get Started**
        ^^^

        Documents in these sections helps you getting started quickly with BigDL-LLM.

        +++
        :bdg-link:`BigDL-LLM in 5 minutes <./Overview/quick_start.html>` |
        :bdg-link:`Installation <./Overview/install.html>`

    .. grid-item-card::

        **Key Features Guide**
        ^^^

        Each guide in this section provides you with in-depth information, concepts and knowledges about BigDL-LLM key features.

        +++

        :bdg-link:`transformers-style <./Overview/KeyFeatures/transformers_style_api.html>` |
        :bdg-link:`LangChain <./Overview/KeyFeatures/langchain_api.html>` |
        :bdg-link:`GPU <./Overview/KeyFeatures/gpu_supports.html>`

    .. grid-item-card::

        **Examples & Tutorials**
        ^^^

        Examples contain scripts to help you quickly get started using BigDL-LLM to run some popular open-source models in the community.

        +++

        :bdg-link:`Examples <./Overview/examples.html>`

    .. grid-item-card::

        **API Document**
        ^^^

        API Document provides detailed description of BigDL-LLM APIs.

        +++

        :bdg-link:`API Document <../PythonAPI/LLM/index.html>`

------

.. raw:: html

    <div>
        <p>
            <sup><a href="#ref-perf" id="footnote-perf">[1]</a>
                Performance varies by use, configuration and other factors. <code><span>bigdl-llm</span></code> may not optimize to the same degree for non-Intel products. Learn more at <a href="http://www.intel.com/PerformanceIndex">www.Intel.com/PerformanceIndex</a>
            </sup>
        </p>
    </div>
                                        
..  toctree::
    :hidden:

    BigDL-LLM Document <self>
