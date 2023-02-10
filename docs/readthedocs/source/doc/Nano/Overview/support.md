# BigDL-Nano Features

| Feature               | Meaning                                                                 |
| --------------------- | ----------------------------------------------------------------------- |
| **Intel-openmp**      | Use Intel-openmp library to improve performance of multithread programs |
| **Jemalloc**          | Use jemalloc as allocator                                               |
| **Tcmalloc**          | Use tcmalloc as allocator                                               |
| **Neural-Compressor** | Neural-Compressor int8 quantization                                     |
| **OpenVINO**          | OpenVINO fp32/bf16/fp16/int8 acceleration on CPU/GPU/VPU                |
| **ONNXRuntime**       | ONNXRuntime fp32/int8 acceleration                                      |
| **CUDA patch**        | Run CUDA code even without GPU                                          |
| **JIT**               | PyTorch JIT optimization                                                |
| **Channel last**      | Channel last memory format                                              |
| **BF16**              | BFloat16 mixed precision training and inference                         |
| **IPEX**              | Intel-extension-for-pytorch optimization                                |
| **Multi-instance**    | Multi-process training and inference                                    |

## Common Feature Support (Can be used in both PyTorch and TensorFlow)

| Feature               | Ubuntu (20.04/22.04) | CentOS7 | MacOS (Intel chip) | MacOS (M-series chip) | Windows |
| --------------------- | -------------------- | ------- | ------------------ | --------------------- | ------- |
| **Intel-openmp**      | ✅                    | ✅       | ✅                  | ②                     | ✅       |
| **Jemalloc**          | ✅                    | ✅       | ✅                  | ❌                     | ❌       |
| **Tcmalloc**          | ✅                    | ❌       | ❌                  | ❌                     | ❌       |
| **Neural-Compressor** | ✅                    | ✅       | ❌                  | ❌                     | ?       |
| **OpenVINO**          | ✅                    | ①       | ❌                  | ❌                     | ④       |
| **ONNXRuntime**       | ✅                    | ①       | ✅                  | ❌                     | ✅       |

## PyTorch Feature Support

| Feature            | Ubuntu (20.04/22.04) | CentOS7 | MacOS (Intel chip) | MacOS (M-series chip) | Windows |
| ------------------ | -------------------- | ------- | ------------------ | --------------------- | ------- |
| **CUDA patch**     | ✅                    | ✅       | ✅                  | ?                     | ✅       |
| **JIT**            | ✅                    | ✅       | ✅                  | ?                     | ✅       |
| **Channel last**   | ✅                    | ✅       | ✅                  | ?                     | ✅       |
| **BF16**           | ✅                    | ✅       | ⭕                  | ⭕                     | ✅       |
| **IPEX**           | ✅                    | ✅       | ❌                  | ❌                     | ❌       |
| **Multi-instance** | ✅                    | ✅       | ②                  | ②                     | ✅       |

## TensorFlow Feature Support

| Feature            | Ubuntu (20.04/22.04) | CentOS7 | MacOS (Intel chip) | MacOS (M-series chip) | Windows |
| ------------------ | -------------------- | ------- | ------------------ | --------------------- | ------- |
| **BF16**           | ✅                    | ✅       | ⭕                  | ⭕                     | ✅       |
| **Multi-instance** | ③                    | ③       | ②③                 | ②③                    | ❌       |

## Symbol Meaning

| Symbol | Meaning                                                                                                  |
| ------ | -------------------------------------------------------------------------------------------------------- |
| ✅      | Supported                                                                                                |
| ❌      | Not supported                                                                                            |
| ⭕      | All Mac machines (Intel/M-series chip) do not support bf16 instruction set, so this feature is pointless |
| ①      | This feature is only supported when used together with jemalloc                                          |
| ②      | This feature is supported but without any performance guarantee                                          |
| ③      | Only Multi-instance training is supported for now                                                        |
| ④      | This feature is only supported when using PyTorch                                                        |
| ?      | Not tested                                                                                               |
