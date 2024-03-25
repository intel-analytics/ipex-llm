IPEX-LLM LangChain API
=====================

LLM Wrapper of LangChain
----------------------------------------

Hugging Face ``transformers`` Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IPEX-LLM provides ``TransformersLLM`` and ``TransformersPipelineLLM``, which implement the standard interface of LLM wrapper of LangChain.

.. tabs::

    .. tab:: AutoModel

        .. automodule:: ipex_llm.langchain.llms.transformersllm
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: model_id, model_kwargs, model, tokenizer, streaming, Config

    .. tab:: pipeline

        .. automodule:: ipex_llm.langchain.llms.transformerspipelinellm
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: pipeline, model_id, model_kwargs, pipeline_kwargs, Config


Native Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ``llama``/``chatglm``/``bloom``/``gptneox``/``starcoder`` model families, you could also use the following LLM wrappers with the native (cpp) implementation for maximum performance.

.. tabs::

    .. tab:: Llama

        .. autoclass:: ipex_llm.langchain.llms.ipexllm.LlamaLLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: ggml_model, ggml_module, client, model_path, kwargs

            .. automethod:: validate_environment
            .. automethod:: stream
            .. automethod:: get_num_tokens

    .. tab:: ChatGLM

        .. autoclass:: ipex_llm.langchain.llms.ipexllm.ChatGLMLLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: ggml_model, ggml_module, client, model_path, kwargs

            .. automethod:: validate_environment
            .. automethod:: stream
            .. automethod:: get_num_tokens

    .. tab:: Bloom

        .. autoclass:: ipex_llm.langchain.llms.ipexllm.BloomLLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: ggml_model, ggml_module, client, model_path, kwargs

            .. automethod:: validate_environment
            .. automethod:: stream
            .. automethod:: get_num_tokens

    .. tab:: Gptneox

        .. autoclass:: ipex_llm.langchain.llms.ipexllm.GptneoxLLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: ggml_model, ggml_module, client, model_path, kwargs

            .. automethod:: validate_environment
            .. automethod:: stream
            .. automethod:: get_num_tokens

    .. tab:: Starcoder

        .. autoclass:: ipex_llm.langchain.llms.ipexllm.StarcoderLLM
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: ggml_model, ggml_module, client, model_path, kwargs

            .. automethod:: validate_environment
            .. automethod:: stream
            .. automethod:: get_num_tokens


Embeddings Wrapper of LangChain
----------------------------------------

Hugging Face ``transformers`` AutoModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ipex_llm.langchain.embeddings.transformersembeddings
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: model, tokenizer, model_id, model_kwargs, encode_kwargs, Config

Native Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ``llama``/``bloom``/``gptneox``/``starcoder`` model families, you could also use the following wrappers.

.. tabs::

    .. tab:: Llama

        .. autoclass:: ipex_llm.langchain.embeddings.ipexllm.LlamaEmbeddings
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: ggml_model, ggml_module, client, model_path, kwargs

            .. automethod:: validate_environment
            .. automethod:: embed_documents
            .. automethod:: embed_query

    .. tab:: Bloom

        .. autoclass:: ipex_llm.langchain.embeddings.ipexllm.BloomEmbeddings
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: ggml_model, ggml_module, client, model_path, kwargs

            .. automethod:: validate_environment
            .. automethod:: embed_documents
            .. automethod:: embed_query

    .. tab:: Gptneox

        .. autoclass:: ipex_llm.langchain.embeddings.ipexllm.GptneoxEmbeddings
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: ggml_model, ggml_module, client, model_path, kwargs

            .. automethod:: validate_environment
            .. automethod:: embed_documents
            .. automethod:: embed_query

    .. tab:: Starcoder

        .. autoclass:: ipex_llm.langchain.embeddings.ipexllm.StarcoderEmbeddings
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: ggml_model, ggml_module, client, model_path, kwargs

            .. automethod:: validate_environment
            .. automethod:: embed_documents
            .. automethod:: embed_query
