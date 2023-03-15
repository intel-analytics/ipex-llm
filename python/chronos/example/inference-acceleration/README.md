# Accelerate the inference speed of model trained on other platform

## Introduction
Chronos has many built-in models wrapped in forecasters, detectors and simulators optimized on CPU (especially intel CPU) platform.

While users may want to use their own model or built-in models trained on another platform (e.g. GPU) but prefer to carry out the inferencing process on CPU platform. Chronos can also help users to accelerate their model for inferencing.

In this example, we show an example to train the model on GPU and accelerate the model by using onnxruntime on CPU.

## How to run this example
```bash
python cpu_inference_acceleration.py
```

More inference acceleration info, please refer to [here](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/speed_up.html#inference-acceleration).

## Sample output
```bash
Epoch 2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 288/288
original pytorch latency (ms): {'p50': 1.984, 'p90': 2.024, 'p95': 2.039, 'p99': 2.461}
onnxruntime latency (ms): {'p50': 0.231, 'p90': 0.24, 'p95': 0.245, 'p99': 0.516}
```