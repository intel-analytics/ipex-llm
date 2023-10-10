BigDL-LLM PyTorch API
=====================

Optimize Model
----------------------------------------

You can run any PyTorch model with one-line ``optimize_model`` to optimize, regardless of the library or API you are using.

.. automodule:: bigdl.llm
    :members: optimize_model
    :undoc-members:
    :show-inheritance:



Load Optimized Model
----------------------------------------

To save model space and speedup loading processes, you can save the model after low-bit optimization and load the saved low-bit model. Saving and loading operations can across various machines, regardless of their operating systems.

.. automodule:: bigdl.llm.optimize
    :members: load_low_bit
    :undoc-members:
    :show-inheritance:
