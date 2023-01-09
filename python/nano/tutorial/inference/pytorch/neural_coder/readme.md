# One-click Accleration Without Code Change

## Introduction
We also provides a no-code method for users to accelerate their pytorch inferencing workflow through neural coder.

## Environment preparation
Neural Compressor >= 2.0 is needed for this function. You may call ``pip install --upgrade neural-compressor`` before using this functionality.

## Run
```bash
# FP32 + Channels_last
python -m neural_coder -o nano_fp32_channels_last example.py
```
Then a new script `example_optimized.py` will be generated and be executed. You may choose other acceleration method name.

## Acceleration Name Table
For `<acceleration_name>`, please check following table.

| Optimization Set | <acceleration_name> | 
| ------------- | ------------- | 
| BF16 + Channels Last | `nano_bf16_channels_last` | 
| BF16 + IPEX + Channels Last | `nano_bf16_ipex_channels_last` | 
| BF16 + IPEX | `nano_bf16_ipex` | 
| BF16 | `nano_bf16` | 
| Channels Last | `nano_fp32_channels_last` | 
| IPEX + Channels Last | `nano_fp32_ipex_channels_last` | 
| IPEX | `nano_fp32_ipex` | 
| INT8 | `nano_int8` | 
| JIT + BF16 + Channels Last | `nano_jit_bf16_channels_last` | 
| JIT + BF16 + IPEX + Channels Last | `nano_jit_bf16_ipex_channels_last` | 
| JIT + BF16 + IPEX | `nano_jit_bf16_ipex` | 
| JIT + BF16 | `nano_jit_bf16` | 
| JIT + Channels Last | `nano_jit_fp32_channels_last` | 
| JIT + IPEX + Channels Last | `nano_jit_fp32_ipex_channels_last` | 
| JIT + IPEX | `nano_jit_fp32_ipex` | 
| JIT | `nano_jit_fp32` | 
| ONNX Runtime | `nano_onnxruntime_fp32` | 
| ONNX Runtime + INT8 | `nano_onnxruntime_int8_qlinear` | 
| OpenVINO | `nano_openvino_fp32` | 
| OpenVINO + INT8 | `nano_openvino_int8` |