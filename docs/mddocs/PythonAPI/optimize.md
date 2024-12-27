# IPEX-LLM PyTorch API

## Optimize Model
You can run any PyTorch model with `optimize_model` through only one-line code change to benefit from IPEX-LLM optimization, regardless of the library or API you are using.

### `ipex_llm.optimize_model`_`(model, low_bit='sym_int4', optimize_llm=True, modules_to_not_convert=None, cpu_embedding=False, **kwargs)`_

A method to optimize any pytorch model.
    
- **Parameters**:

  - **model**: The original PyTorch model (nn.module) 
  
  - **low_bit**: str value, options are `'sym_int4'`, `'asym_int4'`, `'sym_int5'`, `'asym_int5'`, `'sym_int8'`, `'nf3'`, `'nf4'`, `'fp4'`, `'fp8'`, `'fp8_e4m3'`, `'fp8_e5m2'`, `'fp16'` or `'bf16'`, `'sym_int4'` means symmetric int 4, `'asym_int4'` means asymmetric int 4, `'nf4'` means 4-bit NormalFloat, etc. Relevant low bit optimizations will be applied to the model.
  
  - **optimize_llm**: Whether to further optimize llm model. Default to be `True`.
  
  - **modules_to_not_convert**: list of str value, modules (`nn.Module`) that are skipped when conducting model optimizations. Default to be `None`.
  
  - **cpu_embedding**: Whether to replace the Embedding layer, may need to set it to `True` when running IPEX-LLM on GPU. Default to be `False`.
  
- **Returns**: The optimized model.

- **Example**:

  ```python
  # Take OpenAI Whisper model as an example
  from ipex_llm import optimize_model
  model = whisper.load_model('tiny') # Load whisper model under pytorch framework
  model = optimize_model(model) # With only one line code change
  # Use the optimized model without other API change
  result = model.transcribe(audio, verbose=True, language="English")
  # (Optional) you can also save the optimized model by calling 'save_low_bit'
  model.save_low_bit(saved_dir)
  ```

## Load Optimized Model

To avoid high resource consumption during the loading processes of the original model, we provide save/load API to support the saving of model after low-bit optimization and the loading of the saved low-bit model. Saving and loading operations are platform-independent, regardless of their operating systems.

### `ipex_llm.optimize.load_low_bit`_`(model, model_path)`_

Load the optimized pytorch model.

- **Parameters**:

  - **model**: The PyTorch model instance.
  
  - **model_path**: The path of saved optimized model.


- **Returns**: The optimized model.

- **Example**:

  ```python
  # Example 1:
  # Take ChatGLM2-6B model as an example
  # Make sure you have saved the optimized model by calling 'save_low_bit'
  from ipex_llm.optimize import low_memory_init, load_low_bit
  with low_memory_init(): # Fast and low cost by loading model on meta device
      model = AutoModel.from_pretrained(saved_dir,
                                        torch_dtype="auto",
                                        trust_remote_code=True)
  model = load_low_bit(model, saved_dir) # Load the optimized model
  ```

  ```python
  # Example 2:
  # If the model doesn't fit 'low_memory_init' method,
  # alternatively, you can obtain the model instance through traditional loading method.
  # Take OpenAI Whisper model as an example
  # Make sure you have saved the optimized model by calling 'save_low_bit'
  from ipex_llm.optimize import load_low_bit
  model = whisper.load_model('tiny') # A model instance through traditional loading method
  model = load_low_bit(model, saved_dir) # Load the optimized model
  ```
