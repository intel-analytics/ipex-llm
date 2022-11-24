# Use Chronos benchmark tool
This page demonstrates how to use Chronos benchmark tool to benchmark forecasting performance on platforms.

## Basic Usage
The benchmark tool is installed automatically when `bigdl-chronos` is installed. To get information about performance (currently for forecasting only) on the your own machine.

Run benchmark tool with default options using following command:
```bash
benchmark-chronos -l 96 -o 720
```
```eval_rst
.. note::
    **Required Options**:

     ``-l/--lookback`` and ``-o/--horizon`` are required options for Chronos benchmark tool. Use ``-l/--lookback`` to specify the history time steps while use ``-o/--horizon`` to specify the output time steps. For more details, please refer to `here <https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/forecasting.html#regular-regression-rr-style>`_.
```
By default, the tool will load `tsinghua_electricity` dataset and train a `TCNForecaster` with input lookback and horizon parameters under `PyTorch` framework. As it loads, it prints information about hardware, environment variables and benchmark parameters. When benchmarking is completed, it reports the average throughput during training process. Users may be able to improve forecasting performance by following suggested changes on Nano environment variables.

Besides the default usage, more execution parameters can be set to obtain more benchmark results. Read on to learn more about the configuration options available in Chronos benchmark tool.

## Configuration Options
The benchmark tool provides various options for configuring execution parameters. Some key configuration options are introduced in this part and a list of all options is given in [**Advanced Options**](#advanced-options).

### Model
The tool provides several built-in time series forecasting models, including TCN, LSTM, Seq2Seq, NBeats and Autoformer. To specify which model to use, run benchmark tool with `-m/--model`. If not specified, TCN is used as the default.
```bash
benchmark-chronos -m lstm -l 96 -o 720
```

### Stage
Regarding a model, training and inference stages are most concerned. By setting `-s/--stage` parameter, users can obtain knowledge of throughput during training (`-s train`), throughput during inference (`-s throughput`) and latency of inference (`-s latency`). If not specified, train is used as the default.
```bash
benchmark-chronos -s latency -l 96 -o 720
```

### Dataset
Several built-in datasets can be chosen, including nyc_taxi and tsinghua_electricity. If users are with poor Internet connection and hard to download dataset, run benchmark tool with `-d synthetic_dataset` to use synthetic dataset. Default to be tsinghua_electricity if `-d/--dataset` parameter is not specified.
```bash
benchmark-chronos -d nyc_taxi -l 96 -o 720
```
```eval_rst
.. note::
    **Download tsinghua_electricity Dataset**:

     The tsinghua_electricity dataset does not support automatic downloading. Users can download manually from `here <https://github.com/thuml/Autoformer#get-started>`_ to path "~/.chronos/dataset/".
```

### Framework
Pytorch and tensorflow are both supported and can be specified by setting `-f torch` or `-f tensorflow`. And the default framework is pytorch.
```bash
benchmark-chronos -f tensorflow -l 96 -o 720
```
```eval_rst
.. note::
     NBeats and Autoformer does not support tensorflow backend now.
```

### Core number
By default, the benchmark tool will run on all physical cores. And users can explicitly specify the number of cores through `-c/--cores` parameter.
```bash
benchmark-chronos -c 4 -l 96 -o 720
```

### Lookback
Forecasting aims at predicting the future by using the knowledge from the history. The required option `-l/--lookback`corresponds to the length of historical data along time.
```bash
benchmark-chronos -l 96 -o 720
```

### Horizon
Forecasting aims at predicting the future by using the knowledge from the history. The required option `-o/--horizon`corresponds to the length of predicted data along time.
```bash
benchmark-chronos -l 96 -o 720
```

## Advanced Options
Besides, number of processes and epoches can be set by `--training_processes` and `--training_epochs`. Users can also tune batchsize during training and inference through `--training_batchsize` and `--inference_batchsize` respectively.
```bash
benchmark-chronos --training_processes 2 --training_epochs 3 --training_batchsize 32 --inference_batchsize 128 -l 96 -o 720
```

To speed up inference, accelerators like ONNXRuntime and OpenVINO are usually used. To benchmark inference performance with or without accelerator, run tool with `--inference_framework` to specify without accelerator (`--inference_framework torch`)or with ONNXRuntime (`--inference_framework onnx`) or with OpenVINO (`--inference_framework openvino`).
```bash
benchmark-chronos --inference_framework onnx -l 96 -o 720
```

When benchmark tool is run with `--ipex` enabled, intel-extension-for-pytorch will be used as accelerator for trainer. 

If want to use quantized model to predict, just run the benchmark tool with `--quantize` enabled and the quantize framework can be specified by `--quantize_type`. The parameter`--quantize_type` need to be set as pytorch_ipex when users want to use pytorch_ipex as quantize type. Otherwise, the defaut quantize type will be selected according to `--inference_framework`. If pytorch is the inference framework, then pytorch_fx will be the default. If users choose ONNXRuntime as inference framework, onnxrt_qlinearops will be quantize type. And if OpenVINO is chosen, the openvino quantize type will be selected.
```bash
benchmark-chronos --ipex --quantize --quantize_type pytorch_ipex -l 96 -o 720
```


Moreover, if want to benchmark inference performance of a trained model, run benchmark tool with `--ckpt` to specify the checkpoint path of model. By default, the model for inference will be trained first according to input parameters.

Running the benchmark tool with `-h/--help` yields the following usage message, which contains all configuration options:
```bash
benchmark-chronos -h
```
```eval_rst
.. code-block:: python

    usage: benchmark-chronos.py [-h] [-m] [-s] [-d] [-f] [-c] -l lookback -o
                                horizon [--training_processes]
                                [--training_batchsize] [--training_epochs]
                                [--inference_batchsize] [--quantize]
                                [--inference_framework  [...]] [--ipex]
                                [--quantize_type] [--ckpt]

    Benchmarking Parameters

    optional arguments:
      -h, --help            show this help message and exit
      -m, --model           model name, choose from
                            tcn/lstm/seq2seq/nbeats/autoformer, default to "tcn".
      -s, --stage           stage name, choose from train/latency/throughput,
                            default to "train".
      -d, --dataset         dataset name, choose from
                            nyc_taxi/tsinghua_electricity/synthetic_dataset,
                            default to "tsinghua_electricity".
      -f, --framework       framework name, choose from torch/tensorflow, default
                            to "torch".
      -c, --cores           core number, default to all physical cores.
      -l lookback, --lookback lookback
                            required, the history time steps (i.e. lookback).
      -o horizon, --horizon horizon
                            required, the output time steps (i.e. horizon).
      --training_processes 
                            number of processes when training, default to 1.
      --training_batchsize 
                            batch size when training, default to 32.
      --training_epochs     number of epochs when training, default to 1.
      --inference_batchsize 
                            batch size when infering, default to 1.
      --quantize            if use the quantized model to predict, default to
                            False.
      --inference_framework  [ ...]
                            predict without/with accelerator, choose from
                            torch/onnx/openvino, default to "torch" (i.e. predict
                            without accelerator).
      --ipex                if use ipex as accelerator for trainer, default to
                            False.
      --quantize_type       quantize framework, choose from
                            pytorch_fx/pytorch_ipex/onnxrt_qlinearops/openvino,
                            default to "pytorch_fx".
      --ckpt                checkpoint path of a trained model, e.g.
                            "checkpoints/tcn", default to "checkpoints/tcn".
```

