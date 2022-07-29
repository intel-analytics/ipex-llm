# Speed up Chronos built-in models/customized time-series models

Chronos provides transparent acceleration for Chronos built-in models and customized time-series models. In this deep-dive page, we will introduce how to enable/disable them.

We will focus on **single node acceleration for forecasting models' training and inferencing** in this page. Other topic such as:

- Distributed time series data processing - [XShardsTSDataset (based on Spark, powered by `bigdl.orca.data`)](./useful_functionalities.html#xshardstsdataset)
- Distributed training on a cluster - [Distributed training (based on Ray/Spark/Horovod, powered by `bigdl.orca.learn`)](./useful_functionalities.html#distributed-training)
- Non-forecasting models / non-deep-learning models - [Prophet with intel python](./forecasting.html#prophetforecaster), [DBScan Detector with intel Sklearn](./anomaly_detection.html#dbscandetector), [DPGANSimulator pytorch implementation](./simulation.html#dpgansimulator).

You may refer to other pages listed above.

### **1. Overview**
Time series model, especially those deep learning models, often suffers slow training speed and unsatisfying inference speed. Chronos is adapted to integrate many optimized library and best known methods(BKMs) for performance improvement on built-in models and customized models.

### **2. Training Acceleration**
Training Acceleration is transparent in Chronos's API. Transparentness means that Chronos users will enjoy the acceleration without changing their code(unless some expert users want to set some advanced settings).
```eval_rst
.. note:: 
    **Write your script under** ``if __name__=="__main__":``:

     Chronos will automatically utilize the computation resources on the hardware. This may include multi-process training on a single node. Use this header will prevent many strange behavior.
```
#### **2.1 `Forecaster` Training Acceleration**
Currently, transparent acceleration for `LSTMForecaster`, `Seq2SeqForecaster`, `TCNForecaster` and `NBeatsForecaster` is **automatically enabled** and tested. Chronos will set various environment variables and config multi-processing training according to the hardware paremeters(e.g. cores number, ...).

Currently, this function is under active development and **some expert users may want to change some config or disable some acceleration tricks**. Here are some instructions.

Users may unset the environment by:
```bash
source bigdl-nano-unset-env
```
Users may set the the number of process to use in training by:
```python
print(forecaster.num_processes)  # num_processes is automatically optimized by Chronos
forecaster.num_processes = 1  # disable multi-processing training
forecaster.num_processes = 10  # You may set it to any number you want
```
Users may set the IPEX(Intel Pytorch Extension) availbility to use in training by:
```python
print(forecaster.use_ipex)  # use_ipex is automatically optimized by Chronos
forecaster.use_ipex = True  # enable ipex during training
forecaster.use_ipex = False  # disable ipex during training
```

#### **2.2 Customized Model Training Acceleration**
We provide an optimized pytorch-lightning Trainer, `TSTrainer`, to accelerate customized time series model defined by pytorch. A typical use-case can be using `pytorch-forecasting`'s built-in models(they are defined in pytorch-lightning LightningModule) and Chronos `TSTrainer` to accelerate the training process.

`TSTrainer` requires very few code changes to your original code. Here is a quick guide:
```python
# from pytorch-lightning import Trainer
from bigdl.chronos.pytorch import TSTrainer as Trainer

trainer = Trainer(...
                  # set number of processes for training
                  num_processes=8,
                  # disable GPU training, TSTrainer currently only available for CPU
                  gpus=0,
                  ...)
```

We have examples adapted from `pytorch-forecasting`'s examples to show the significant speed-up by using `TSTrainer` in our [use-case](https://github.com/intel-analytics/BigDL/tree/main/python/chronos/use-case/pytorch-forecasting).

#### **2.3 Auto Tuning Acceleration**
We are working on the acceleration of `AutoModel` and `AutoTSEstimator`. Please unset the environment by:
```bash
source bigdl-nano-unset-env
``` 

### **3. Inference Acceleration**
Inference has become a critical part for time series model's performance. This may be divided to two parts:
- Throughput: how many samples can be predicted in a certain amount of time.
- Latency: how much time is used to predict 1 sample.

Typically, throughput and latency is a trade-off pair. We have three optimization options for inferencing in Chronos.
- **Default**: Generally useful for both throughput and latency.
- **ONNX Runtime**: Users may export their trained(w/wo auto tuning) model to ONNX file and deploy it on other service. Chronos also provides an internal onnxruntime inference support for those users who pursue low latency and higher throughput during inference on a single node.
- **Quantization**: Quantization refers to processes that enable lower precision inference. In Chronos, post-training quantization is supported relied on [Intel® Neural Compressor](https://intel.github.io/neural-compressor/README.html).
```eval_rst
.. note:: 
    **Additional Dependencies**:

    You need to install `neural-compressor` to enable quantization related methods.

    ``pip install neural-compressor==1.8.1``
```
#### **3.1 `Forecaster` Inference Acceleration**
##### **3.1.1 Default Acceleration**
Nothing needs to be done. Chronos has deployed accleration for inferencing. **some expert users may want to change some config or disable some acceleration tricks**. Here are some instructions:

Users may unset the environment by:
```bash
source bigdl-nano-unset-env
```
##### **3.1.2 ONNX Runtime**
LSTM, TCN, Seq2seq and NBeats has supported onnx in their forecasters. When users use these built-in models, they may call `predict_with_onnx`/`evaluate_with_onnx` for prediction or evaluation. They may also call `export_onnx_file` to export the onnx model file and `build_onnx` to change the onnxruntime's setting(not necessary).
```python
f = Forecaster(...)
f.fit(...)
f.predict_with_onnx(...)
```
##### **3.1.3 Quantization**
LSTM, TCN and NBeats has supported quantization in their forecasters.
```python
# init
f = Forecaster(...)

# train the forecaster
f.fit(train_data, ...)

# quantize the forecaster
f.quantize(train_data, ..., framework=...)

# predict with int8 model with better inference throughput
f.predict/predict_with_onnx(test_data, quantize=True)

# predict with fp32
f.predict/predict_with_onnx(test_data, quantize=False)

# save
f.save(checkpoint_file="fp32.model"
       quantize_checkpoint_file="int8.model")

# load
f.load(checkpoint_file="fp32.model"
       quantize_checkpoint_file="int8.model")
```
Please refer to [Forecaster API Docs](../../PythonAPI/Chronos/forecasters.html) for details.

#### **3.2 `TSPipeline` Inference Acceleration**
Basically same to [`Forecaster`](#31-forecaster-inference-acceleration)
##### **3.1.1 Default Acceleration**
Basically same to [`Forecaster`](#31-forecaster-inference-acceleration)
##### **3.1.2 ONNX Runtime**
```python
tsppl.predict_with_onnx(...)
```
##### **3.1.3 Quantization**
```python
tsppl.quantize(...)
tsppl.predict/predict_with_onnx(test_data, quantize=True/False)
```
Please refer to [TSPipeline API doc](../../PythonAPI/Chronos/autotsestimator.html#tspipeline) for details.