# Accelerate the inference speed of model trained on other platform

## Introduction
Chronos has many built-in models wrapped in forecasters, detectors and simulators optimized on CPU (especially intel CPU) platform.

While users may want to use their own model or built-in models trained on another platform (e.g. GPU) but prefer to carry out the inferencing process on CPU platform. Chronos can also help users to accelerate their model for inferencing.

In this example, we show an example to train the model on GPU and accelerate the model by using onnxruntime on CPU.

## How to run this example
```bash
python cpu_inference_acceleration.py
```

## Sample output
```bash
Epoch 2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 288/288
original pytorch latency (ms): {'p50': 1.236, 'p90': 1.472, 'p95': 1.612, 'p99': 32.989}
onnxruntime latency (ms): {'p50': 0.124, 'p90': 0.129, 'p95': 0.148, 'p99': 0.363}
```