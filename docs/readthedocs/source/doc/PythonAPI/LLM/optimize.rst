BigDL-LLM PyTorch API
=====================

Optimize Model
----------------------------------------

You can run any PyTorch model with ``optimize_model`` through only one-line code change to benefit from BigDL-LLM optimization, regardless of the library or API you are using.

.. automodule:: ipex_llm
    :members: optimize_model
    :undoc-members:
    :show-inheritance:



Load Optimized Model
----------------------------------------

To avoid high resource consumption during the loading processes of the original model, we provide save/load API to support the saving of model after low-bit optimization and the loading of the saved low-bit model. Saving and loading operations are platform-independent, regardless of their operating systems.

.. automodule:: ipex_llm.optimize
    :members: load_low_bit
    :undoc-members:
    :show-inheritance:
