BigDL-LLM ``transformers``-style API
====================================

Hugging Face ``transformers`` Format
------------------------------------

You can apply BigDL-LLM optimizations on any Hugging Face Transformers models by using the standard AutoModel APIs.


AutoModelForCausalLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: bigdl.llm.transformers.model.AutoModelForCausalLM
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: from_pretrained
    .. automethod:: load_convert
    .. automethod:: load_low_bit

AutoModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: bigdl.llm.transformers.model.AutoModel
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: from_pretrained
    .. automethod:: load_convert
    .. automethod:: load_low_bit

AutoModelForSpeechSeq2Seq
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: bigdl.llm.transformers.model.AutoModelForSpeechSeq2Seq
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: from_pretrained
    .. automethod:: load_convert
    .. automethod:: load_low_bit

AutoModelForSeq2SeqLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: bigdl.llm.transformers.model.AutoModelForSeq2SeqLM
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: from_pretrained
    .. automethod:: load_convert
    .. automethod:: load_low_bit



Native Format
----------------------------------------

For ``llama``/``chatglm``/``bloom``/``gptneox``/``starcoder`` model families, you may also convert and run LLM using the native (cpp) implementation for maximum performance.


.. tabs::

    .. tab:: Llama

        .. autoclass:: bigdl.llm.transformers.modelling_bigdl.LlamaForCausalLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: GGML_Model, GGML_Module, HF_Class

            .. automethod:: from_pretrained

    .. tab:: ChatGLM

        .. autoclass:: bigdl.llm.transformers.modelling_bigdl.ChatGLMForCausalLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: GGML_Model, GGML_Module, HF_Class

            .. automethod:: from_pretrained

    .. tab:: Gptneox

        .. autoclass:: bigdl.llm.transformers.modelling_bigdl.GptneoxForCausalLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: GGML_Model, GGML_Module, HF_Class

            .. automethod:: from_pretrained

    .. tab:: Bloom
        .. autoclass:: bigdl.llm.transformers.modelling_bigdl.BloomForCausalLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: GGML_Model, GGML_Module, HF_Class    

            .. automethod:: from_pretrained

    .. tab:: Starcoder

        .. autoclass:: bigdl.llm.transformers.modelling_bigdl.StarcoderForCausalLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: GGML_Model, GGML_Module, HF_Class

            .. automethod:: from_pretrained
