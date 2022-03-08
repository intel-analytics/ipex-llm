# Use Chronos forecasters' quantization method to speed-up inference
LSTM, TCN and NBeats users can easily quantize their forecasters to low precision and speed up the inference process (both throughput and latency) by on a single node. The functionality is powered by Project Nano.

## Prepare the environment
We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:
```bash
conda create -n my_env python=3.7 # "my_env" is conda environment name, you can use any name you like.
conda activate my_env
pip install --pre --upgrade bigdl-chronos[all]
pip install neural-compressor==1.8.1
```
Please refer to [Chronos Install Guide](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html#install)

## Prepare data
We are using the nyc taxi provided by NAB, from 2014-07-01 to 2015-01-31 taxi fare information For more details, you can refer to the detailed information [here](https://github.com/numenta/NAB/tree/master/data). The dataset will be downloaded automatically for you.

## Run the example
For tcn forecaster example
```bash
python quantization_tcnforecaster_nyc_taxi.py
```

## Sample output
In this example, we perform post-training quantization on pytorch model for inference throughput and on onnxruntime model for inference latency. Both of them are significantly improved by 339.5% and 89.4%.
```bash
# ... <skip some training progress bar>
# ... <skip some Intel Neural Compressor log>
Pytorch Quantization helps increase inference throughput by 339.49 %
Onnx Quantization helps decrease inference latency by 89.39 %
fp32 pytorch smape: 0.21
int8 pytorch smape: 0.21
int8 onnx smape: 0.21
```