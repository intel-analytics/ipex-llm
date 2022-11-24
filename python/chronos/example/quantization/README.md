# Use Chronos forecasters' quantization method to speed-up inference
LSTM, TCN and NBeats users can easily quantize their forecasters to low precision and speed up the inference process (both throughput and latency) by on a single node. The functionality is powered by Project Nano.

## Prepare the environment
We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster.
Please refer to [Chronos Install Guide](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/install.html)

For this example your may just
```bash
pip install --pre --upgrade bigdl-chronos[pytorch,inference]
```

## Prepare data
We are using the nyc taxi provided by NAB, from 2014-07-01 to 2015-01-31 taxi fare information For more details, you can refer to the detailed information [here](https://github.com/numenta/NAB/tree/master/data). The dataset will be downloaded automatically for you.

## Run the example
For tcn forecaster example
```bash
taskset -c 0-7 python quantization_tcnforecaster_nyc_taxi.py
```

Here we use `taskset -c 0-7` to limit the core usage to first 8 cores and make the output more stable.

## Sample output
In this example, we perform post-training quantization on pytorch model for inference throughput and on onnxruntime model for inference latency. Both of them are significantly improved by 21.75 % and 89.4%.
```bash
# ... <skip some training progress bar>
# ... <skip some Intel Neural Compressor log>
Pytorch Quantization helps increase inference throughput by 71.89 %
Onnx Quantization helps decrease inference latency by 89.99 %
fp32 pytorch smape: 16.6
int8 pytorch smape: 16.58
int8 onnx smape: 16.58
```

## Performance Tuning
Quantization trick will transform the weight of the model to int8 type to improve the performance on CPU and utilize (if possible) high throughput instruction set such as VNNI.

Quantization typically has a good throughput when `batch_size`, `lookback`, `horizon` and the hyperparameters of the model are large.
