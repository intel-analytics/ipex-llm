# Chronos User Guide

### **1. Overview**
_BigDL-Chronos_ (_Chronos_ for short) is an application framework for building a fast, accurate and scalable time series analysis application.

You can use _Chronos_ to do:

- **Data pre/post-processing and feature generation** (using [TSDataset](./data_processing_feature_engineering.html))
- **Time Series Forecasting** (using [Standalone Forecasters](./forecasting.html#use-standalone-forecaster-pipeline), [Auto Models](./forecasting.html#use-auto-forecasting-model) (with HPO) or [AutoTS](./forecasting.html#use-autots-pipeline) (full AutoML enabled pipelines))
- **Anomaly Detection** (using [Anomaly Detectors](./anomaly_detection.html#anomaly-detection))
- **Synthetic Data Generation** (using [Simulators](./simulation.html#generate-synthetic-data))
- **Speed up or tune your customized time-series model** (using TSTrainer and [AutoTS](./forecasting.html#use-autots-pipeline))

---
### **2. Install**

```eval_rst
.. raw:: html

    <link rel="stylesheet" type="text/css" href="../../../_static/css/chronos_installation_guide.css" />

    <div class="displayed">
        <table id="table-1" style="margin:auto">
            <thead>
                <th>AI Framework</th>
                <th colspan="1"><button id="pytorch"
                        title="Use PyTorch as deep learning models' backend. Most of the model support and works better under PyTorch.">PyTorch</br>(Recommended)</button>
                </th>
                <th colspan="1"><button id="tensorflow"
                        title="Use Tensorflow as deep learning models' backend.">Tensorflow</button></th>
                <th colspan="1"><button id="prophet" title="For Prophet model.">Prophet</button></th>
                <th colspan="1"><button id="pmdarima" title="For ARIMA model.">ARIMA</button></th>
            </thead>
            <tbody>

                <tr>
                    <td>OS</td>
                    <td colspan="2"><button id="linux" title="Ubuntu/CentOS is recommended">Linux</button></td>
                    <td colspan="2"><button id="win" title="WSL is needed for Windows users">Windows</button></td>
                </tr>

                <tr>
                    <td>Auto Tuning</td>
                    <td colspan="2" title="I don't need any hyperparameter auto tuning feature."><button
                            id="automlno">No need</button></td>
                    <td colspan="2" title="I need chronos to help me tune the hyperparameters."><button
                            id="automlyes">Needed</button></td>
                </tr>


                <tr>
                    <td>Hardware</td>
                    <td colspan="2"><button id="singlenode" title="For users use laptop/single node server.">Single
                            node</button></td>
                    <td colspan="2"><button id="cluster" title="For users use K8S/Yarn Cluster.">Cluster</button></td>
                </tr>

                <tr>
                    <td>Release</td>
                    <td colspan="2"><button id="pypi" title="For users use pip to install chronos.">Pip</button></td>
                    <td colspan="2"><button id="docker" title="For users use docker image.">Docker</button></td>
                </tr>

                <tr>
                    <td>Build</td>
                    <td colspan="2"><button id="stable"
                            title="For users would like to deploy chronos in their production">Stable (2.0.0)</button>
                    </td>
                    <td colspan="2"><button id="nightly"
                            title="For users would like to try chronos's latest feature">Nightly (2.1.0b)</button></td>
                </tr>

                <tr>
                    <td>Install CMD</td>
                    <td colspan="4">
                        <div id="cmd" style="text-align: left;">NA</div>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>

    <script src="../../../_static/js/chronos_installation_guide.js"></script> 
```

</br>

#### **2.1 Pypi**
When you install `bigdl-chronos` from PyPI. We recommend to install with a conda virtual environment. To install Conda, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).
```bash
conda create -n my_env python=3.7 setuptools=58.0.4
conda activate my_env
pip install --pre --upgrade bigdl-chronos[pytorch]  # or other options you may want to use
source bigdl-nano-init
```
#### **2.2 Tensorflow backend**
Tensorflow is one of the supported backend of Chronos in nightly release version, while it can not work alone without pytorch in Chronos for now. We will fix it soon. If you want to use tensorflow backend, please
```bash
pip install --pre --upgrade bigdl-nano[tensorflow]
```
after you install the pytorch backend chronos.

#### **2.3 OS and Python version requirement**

```eval_rst
.. note:: 
    **Supported OS**:

     Chronos is thoroughly tested on Ubuntu (16.04/18.04/20.04), and should works fine on CentOS. If you are a Windows user, the most convenient way to use Chronos on a windows laptop might be using WSL2, you may refer to https://docs.microsoft.com/en-us/windows/wsl/setup/environment or just install a ubuntu virtual machine.
```
```eval_rst
.. note:: 
    **Supported Python Version**:

     Chronos only supports Python 3.7.2 ~ latest 3.7.x. We are validating more Python versions.
```

---


### **3. Which document to see?**

```eval_rst
.. grid:: 2
    :gutter: 1

    .. grid-item-card::
        :class-footer: sd-bg-light

        **Quick Tour**
        ^^^

        You may understand the basic usage of Chronos' components and learn to write the first runnable application in this quick tour page.

        +++
        `Quick Tour <./quick-tour.html>`_

    .. grid-item-card::
        :class-footer: sd-bg-light

        **User Guides**
        ^^^

        Our user guides provide you with in-depth information, concepts and knowledges about Chronos.

        +++

        `Data process <./data_processing_feature_engineering.html>`_ / 
        `Forecast <./forecasting.html>`_ / 
        `Detect <./anomaly_detection.html>`_ / 
        `Simulate <./simulation.html>`_

.. grid:: 2
    :gutter: 1

    .. grid-item-card::
        :class-footer: sd-bg-light

        **How-to-Guide**
        ^^^

        If you are meeting with some specific problems during the usage, how-to guides are good place to be checked.

        +++

        Work In Progress

    .. grid-item-card::
        :class-footer: sd-bg-light

        **API Document**
        ^^^

        API Document provides you with a detailed description of the Chronos APIs. 

        +++

        `API Document <../../PythonAPI/Chronos/index.html>`_

```

---

### **4. Examples and Demos**
- Quickstarts
    - [Use AutoTSEstimator for Time-Series Forecasting](../QuickStart/chronos-autotsest-quickstart.html)
    - [Use TSDataset and Forecaster for Time-Series Forecasting](../QuickStart/chronos-tsdataset-forecaster-quickstart.html)
    - [Use Anomaly Detector for Unsupervised Anomaly Detection](../QuickStart/chronos-anomaly-detector.html)
- Examples
    - [Use AutoLSTM on nyc taxi dataset][autolstm_nyc_taxi]
    - [Use AutoProphet on nyc taxi dataset][autoprophet_nyc_taxi]
    - [High dimension time series forecasting with Chronos TCMFForecaster][run_electricity]
    - [Use distributed training with Chronos Seq2SeqForecaster][distributed_training_network_traffic]
    - [Use ONNXRuntime to accelerate the inference of AutoTSEstimator][onnx_autotsestimator_nyc_taxi]
    - [Use ONNXRuntime to accelerate the inference of Seq2SeqForecaster][onnx_forecaster_network_traffic]
    - [Generate synthetic data with DPGANSimulator in a data-driven fashion][simulator]
    - [Quantizate your forecaster to speed up inference][quantization]
- Use cases
    - [Unsupervised Anomaly Detection][AIOps_anomaly_detect_unsupervised]
    - [Unsupervised Anomaly Detection based on Forecasts][AIOps_anomaly_detect_unsupervised_forecast_based]
    - [Stock Price Prediction with LSTM][stock_prediction]
    - [Stock Price Prediction with ProphetForecaster and AutoProphet][stock_prediction_prophet]
    - [Network Traffic Forecasting with AutoTSEstimator][network_traffic_autots_forecasting]
    - [Network Traffic Forecasting (using multivariate time series data)][network_traffic_model_forecasting]
    - [Network Traffic Forecasting (using multistep time series data)][network_traffic_multivariate_multistep_tcnforecaster]
    - [Network Traffic Forecasting with Customized Model][network_traffic_autots_customized_model]
    - [Help pytorch-forecasting improve the training speed of DeepAR model][pytorch_forecasting_deepar]
    - [Help pytorch-forecasting improve the training speed of TFT model][pytorch_forecasting_tft]

<!--Reference links in article-->
[autolstm_nyc_taxi]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/example/auto_model/autolstm_nyc_taxi.py>
[autoprophet_nyc_taxi]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/example/auto_model/autoprophet_nyc_taxi.py>
[run_electricity]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/example/tcmf/run_electricity.py>
[distributed_training_network_traffic]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/example/distributed/distributed_training_network_traffic.py>
[onnx_autotsestimator_nyc_taxi]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/example/onnx/onnx_autotsestimator_nyc_taxi.py>
[onnx_forecaster_network_traffic]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/example/onnx/onnx_forecaster_network_traffic.py>
[simulator]: <https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/simulator>
[AIOps_anomaly_detect_unsupervised]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.ipynb>
[AIOps_anomaly_detect_unsupervised_forecast_based]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.ipynb>
[stock_prediction]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/fsi/stock_prediction.ipynb>
[stock_prediction_prophet]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/fsi/stock_prediction_prophet.ipynb>
[network_traffic_autots_forecasting]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting.ipynb>
[network_traffic_model_forecasting]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_model_forecasting.ipynb>
[network_traffic_multivariate_multistep_tcnforecaster]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.ipynb>
[network_traffic_autots_customized_model]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model.ipynb>
[quantization]: <https://github.com/intel-analytics/BigDL/blob/main/python/chronos/example/quantization/quantization_tcnforecaster_nyc_taxi.py>
[pytorch_forecasting_deepar]: <https://github.com/intel-analytics/BigDL/tree/main/python/chronos/use-case/pytorch-forecasting/DeepAR>
[pytorch_forecasting_tft]: <https://github.com/intel-analytics/BigDL/tree/main/python/chronos/use-case/pytorch-forecasting/TFT>
