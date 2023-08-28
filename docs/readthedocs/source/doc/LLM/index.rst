BigDL-LLM
=========================


**BigDL-LLM** is a library for running **LLM** (Large Language Models) on your Intel **laptop** using INT4 with very low latency [*]_ (for any Hugging Face *Transformers* model).

See the **optimized performance** of ``chatglm2-6b``, ``llama-2-13b-chat``, and ``starcoder-15b`` models on a 12th Gen Intel Core CPU below.

.. grid:: 1 3 3 3
    :gutter: 2

    .. grid-item::
        :class: sd-text-center

        .. image:: https://github.com/bigdl-project/bigdl-project.github.io/raw/master/assets/chatglm2-6b.gif
        
        chatglm2-6b

    .. grid-item::
        :class: sd-text-center

        .. image:: https://github.com/bigdl-project/bigdl-project.github.io/raw/master/assets/llama-2-13b-chat.gif
        
        llama-2-13b-chat

    .. grid-item::
        :class: sd-text-center

        .. image:: https://github.com/bigdl-project/bigdl-project.github.io/raw/master/assets/llm-15b5.gif
        
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

        :bdg-link:`transformers-style API <./Overview/KeyFeatures/transformers_style_api.html>` |
        :bdg-link:`LangChain API <./Overview/KeyFeatures/langchain_api.html>` |
        :bdg-link:`CLI <./Overview/KeyFeatures/cli.html>`

    .. grid-item-card::

        **Examples & Tutorials**
        ^^^

        Model Supports contain example scripts to help you quickly get started using BigDL-LLM to run some popular open-source models in the community.
        Tutorials will provide you in-depth hands-on experience for BigDL-LLM main usage.

        +++

        :bdg-link:`Model Supports <./Overview/model_supports.html>`

    .. grid-item-card::

        **API Document**
        ^^^

        API Document provides detailed description of BigDL-LLM APIs.

        +++

        :bdg-link:`API Document <../PythonAPI/LLM/index.html>`

------

.. [*] Performance varies by use, configuration and other factors. ``bigdl-llm`` may not optimize to the same degree for non-Intel products. Learn more at `www.Intel.com/PerformanceIndex <http://www.intel.com/PerformanceIndex>`_.

..  toctree::
    :hidden:

    BigDL-LLM Document <self>
