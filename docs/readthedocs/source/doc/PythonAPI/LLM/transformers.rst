BigDL-LLM ``transformers``-style API
====================================

Hugging Face ``transformers`` AutoModel
------------------------------------

You can apply BigDL-LLM optimizations on any Hugging Face Transformers models by using the standard AutoModel APIs.


AutoModelForCausalLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ipex_llm.transformers.AutoModelForCausalLM
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: from_pretrained
    .. automethod:: load_convert
    .. automethod:: load_low_bit

AutoModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ipex_llm.transformers.AutoModel
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: from_pretrained
    .. automethod:: load_convert
    .. automethod:: load_low_bit

AutoModelForSpeechSeq2Seq
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ipex_llm.transformers.AutoModelForSpeechSeq2Seq
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: from_pretrained
    .. automethod:: load_convert
    .. automethod:: load_low_bit

AutoModelForSeq2SeqLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ipex_llm.transformers.AutoModelForSeq2SeqLM
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: from_pretrained
    .. automethod:: load_convert
    .. automethod:: load_low_bit



Native Model
----------------------------------------

For ``llama``/``chatglm``/``bloom``/``gptneox``/``starcoder`` model families, you may also convert and run LLM using the native (cpp) implementation for maximum performance.


.. tabs::

    .. tab:: Llama

        .. autoclass:: ipex_llm.transformers.LlamaForCausalLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: GGML_Model, GGML_Module, HF_Class

            .. automethod:: from_pretrained

    .. tab:: ChatGLM

        .. autoclass:: ipex_llm.transformers.ChatGLMForCausalLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: GGML_Model, GGML_Module, HF_Class

            .. automethod:: from_pretrained

    .. tab:: Gptneox

        .. autoclass:: ipex_llm.transformers.GptneoxForCausalLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: GGML_Model, GGML_Module, HF_Class

            .. automethod:: from_pretrained

    .. tab:: Bloom
        .. autoclass:: ipex_llm.transformers.BloomForCausalLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: GGML_Model, GGML_Module, HF_Class    

            .. automethod:: from_pretrained

    .. tab:: Starcoder

        .. autoclass:: ipex_llm.transformers.StarcoderForCausalLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: GGML_Model, GGML_Module, HF_Class

            .. automethod:: from_pretrained
