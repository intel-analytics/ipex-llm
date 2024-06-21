# Multi Intel GPUs selection

In [Inference on GPU](inference_on_gpu.md) and [Finetune (QLoRA)](finetune.md), you have known how to run inference and finetune on Intel GPUs. In this section, we will show you two approaches to select GPU devices.

## List devices

The `sycl-ls` tool enumerates a list of devices available in the system. You can use it after you setup oneapi environment:

- For **Windows users**:

   Please make sure you are using CMD (Miniforge Prompt if using conda):

   ```cmd
   call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
   sycl-ls
   ```

- For **Linux users**:

   ```bash
   source /opt/intel/oneapi/setvars.sh
   sycl-ls
   ```


If you have two Arc770 GPUs, you can get something like below:
```
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.16.7.0.21_160000]
[opencl:cpu:1] Intel(R) OpenCL, Intel(R) Core(TM) i9-14900K 3.0 [2023.16.7.0.21_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[opencl:gpu:3] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[opencl:gpu:4] Intel(R) OpenCL Graphics, Intel(R) UHD Graphics 770 3.0 [23.17.26241.33]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26241]
[ext_oneapi_level_zero:gpu:1] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26241]
[ext_oneapi_level_zero:gpu:2] Intel(R) Level-Zero, Intel(R) UHD Graphics 770 1.3 [1.3.26241]
```
This output shows there are two Arc A770 GPUs as well as an Intel iGPU on this machine.

## Devices selection
To enable xpu, you should convert your model and input to xpu by below code:
```python
model = model.to('xpu')
input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
```
To select the desired devices, there are two ways: one is changing the code, another is adding an environment variable. See:  

### 1. Select device in python
To specify a xpu, you can change the `to('xpu')` to `to('xpu:[device_id]')`, this device_id is counted from zero.

If you you want to use the second device, you can change the code like this: 
```python
model = model.to('xpu:1')
input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu:1')
```

### 2. OneAPI device selector
Device selection environment variable, `ONEAPI_DEVICE_SELECTOR`, can be used to limit the choice of Intel GPU devices. As upon `sycl-ls` shows, the last three lines are three Level Zero GPU devices. So we can use `ONEAPI_DEVICE_SELECTOR=level_zero:[gpu_id]` to select devices.
For example, you want to use the second A770 GPU, you can run the python like this:

- For **Windows users**:

   ```cmd
   set ONEAPI_DEVICE_SELECTOR=level_zero:1 
   python generate.py
   ```
   Through ``set ONEAPI_DEVICE_SELECTOR=level_zero:1``, only the second A770 GPU will be available for the current environment.

- For **Linux users**:

   ```bash
   ONEAPI_DEVICE_SELECTOR=level_zero:1 python generate.py
   ```

   ``ONEAPI_DEVICE_SELECTOR=level_zero:1`` in upon command only affect in current python program. Also, you can export the environment variable, then run your python:

   ```bash
   export ONEAPI_DEVICE_SELECTOR=level_zero:1
   python generate.py
   ```
