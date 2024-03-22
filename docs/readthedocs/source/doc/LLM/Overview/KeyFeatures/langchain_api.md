# LangChain API

You may run the models using the LangChain API in `bigdl-llm`.

## Using Hugging Face `transformers` INT4 Format

You may run any Hugging Face *Transformers* model (with INT4 optimiztions applied) using the LangChain API as follows:

```python
from ipex_llm.langchain.llms import TransformersLLM
from ipex_llm.langchain.embeddings import TransformersEmbeddings
from langchain.chains.question_answering import load_qa_chain

embeddings = TransformersEmbeddings.from_model_id(model_id=model_path)
bigdl_llm = TransformersLLM.from_model_id(model_id=model_path, ...)

doc_chain = load_qa_chain(bigdl_llm, ...)
output = doc_chain.run(...)
```

```eval_rst
.. seealso::

   See the examples `here <https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/CPU/LangChain/transformers_int4>`_.
```

## Using Native INT4 Format

You may also convert Hugging Face *Transformers* models into native INT4 format, and then run the converted models using the LangChain API as follows.

```eval_rst
.. note::

   * Currently only llama/bloom/gptneox/starcoder/chatglm model families are supported; for other models, you may use the Hugging Face ``transformers`` INT4 format as described `above <./langchain_api.html#using-hugging-face-transformers-int4-format>`_.

   * You may choose the corresponding API developed for specific native models to load the converted model.
```

```python
from ipex_llm.langchain.llms import LlamaLLM
from ipex_llm.langchain.embeddings import LlamaEmbeddings
from langchain.chains.question_answering import load_qa_chain

# switch to ChatGLMEmbeddings/GptneoxEmbeddings/BloomEmbeddings/StarcoderEmbeddings to load other models
embeddings = LlamaEmbeddings(model_path='/path/to/converted/model.bin')
# switch to ChatGLMLLM/GptneoxLLM/BloomLLM/StarcoderLLM to load other models
bigdl_llm = LlamaLLM(model_path='/path/to/converted/model.bin')

doc_chain = load_qa_chain(bigdl_llm, ...)
doc_chain.run(...)
```

```eval_rst
.. seealso::

   See the examples `here <https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/CPU/LangChain/native_int4>`_.
```
