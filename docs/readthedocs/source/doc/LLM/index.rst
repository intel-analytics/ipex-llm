IPEX-LLM
=========================

.. raw:: html

   <p>
      <a href="https://github.com/intel-analytics/ipex-llm/tree/main/python/llm"><code><span>ipex-llm</span></code></a> is a library for running <strong>LLM</strong> (large language model) on Intel <strong>XPU</strong> (from <em>Laptop</em> to <em>GPU</em> to <em>Cloud</em>) using <strong>INT4</strong> with very low latency <sup><a href="#footnote-perf" id="ref-perf">[1]</a></sup> (for any <strong>PyTorch</strong> model).
   </p>

-------

.. grid:: 1 2 2 2
    :gutter: 2

    .. grid-item-card::

        **Get Started**
        ^^^

        Documents in these sections helps you getting started quickly with IPEX-LLM.

        +++
        :bdg-link:`IPEX-LLM in 5 minutes <./Overview/llm.html>` |
        :bdg-link:`Installation <./Overview/install.html>`

    .. grid-item-card::

        **Key Features Guide**
        ^^^

        Each guide in this section provides you with in-depth information, concepts and knowledges about IPEX-LLM key features.

        +++

        :bdg-link:`PyTorch <./Overview/KeyFeatures/optimize_model.html>` |
        :bdg-link:`transformers-style <./Overview/KeyFeatures/transformers_style_api.html>` |
        :bdg-link:`LangChain <./Overview/KeyFeatures/langchain_api.html>` |
        :bdg-link:`GPU <./Overview/KeyFeatures/gpu_supports.html>`

    .. grid-item-card::

        **Examples & Tutorials**
        ^^^

        Examples contain scripts to help you quickly get started using IPEX-LLM to run some popular open-source models in the community.

        +++

        :bdg-link:`Examples <./Overview/examples.html>`

    .. grid-item-card::

        **API Document**
        ^^^

        API Document provides detailed description of IPEX-LLM APIs.

        +++

        :bdg-link:`API Document <../PythonAPI/LLM/index.html>`

------

.. raw:: html

    <div>
        <p>
            <sup><a href="#ref-perf" id="footnote-perf">[1]</a>
               Performance varies by use, configuration and other factors. <code><span>ipex-llm</span></code> may not optimize to the same degree for non-Intel products. Learn more at <a href="https://www.Intel.com/PerformanceIndex">www.Intel.com/PerformanceIndex</a>.
            </sup>
        </p>
    </div>
                                        
..  toctree::
    :hidden:

    IPEX-LLM Document <self>
