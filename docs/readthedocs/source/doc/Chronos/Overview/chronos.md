# Chronos User Guide

### **1. Overview**
_Chronos_ is an application framework for building large-scale time series analysis applications.

You can use _Chronos_ to do:

- **Data pre/post-processing and feature generation** (using [TSDataset](./data_processing_feature_engineering.html))
- **Time Series Forecasting** (using [Standalone Forecasters](./forecasting.html#use-standalone-forecaster-pipeline), [Auto Models](./forecasting.html#use-auto-forecasting-model) (with HPO) or [AutoTS](./forecasting.html#use-autots-pipeline) (full AutoML enabled pipelines))
- **Anomaly Detection** (using [Anomaly Detectors](./anomaly_detection.html#anomaly-detection))
- **Synthetic Data Generation** (using [Simulators](./simulation.html#generate-synthetic-data))

---
### **2. Install**

Install `bigdl-chronos` from PyPI. We recommened to install with a conda virtual environment.
```bash
conda create -n my_env python=3.7
conda activate my_env
pip install bigdl-chronos
```
You may also install `bigdl-chronos` with target `[all]` to install the additional dependencies for _Chronos_. This will enable distributed tuning with AutoTS.
```bash
# stable version
pip install bigdl-chronos[all]
# nightly built version
pip install --pre --upgrade bigdl-chronos[all]
```
---
### **3. Run**
Various python programming environments are supported to run a _Chronos_ application.

#### **3.1 Jupyter Notebook**

You can start the Jupyter notebook as you normally do using the following command and run  _Chronos_ application directly in a Jupyter notebook:

```bash
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```

#### **3.2 Python Script**

You can directly write _Chronos_ application in a python file (e.g. script.py) and run in the command line as a normal Python program:

```bash
python script.py
```

---
### **4. Get Started**

#### **4.1 Initialization**
_Chronos_ uses [Orca](../../Orca/Overview/orca.md) to enable distributed training and AutoML capabilities. Initialize orca as below when you want to:

1. Use the distributed mode of a forecaster.
2. Use automl to distributedly tuning your model.
3. Use `XshardsTSDataset` to process time series dataset in distribution fashion.

Otherwise, there is no need to initialize an orca context.

View [Orca Context](../../Orca/Overview/orca-context.md) for more details. Note that argument `init_ray_on_spark` must be `True` for _Chronos_. 

```python
from bigdl.orca.common import init_orca_context, stop_orca_context

# run in local mode
init_orca_context(cluster_mode="local", cores=4, init_ray_on_spark=True)
# run on K8s cluster
init_orca_context(cluster_mode="k8s", num_nodes=2, cores=2, init_ray_on_spark=True)
# run on Hadoop YARN cluster
init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2, init_ray_on_spark=True)

# >>> Start of Chronos Application >>>
# ...
# <<< End of Chronos Application <<<

stop_orca_context()
```
#### **4.2 AutoTS Example**

This example run a forecasting task with automl optimization with `AutoTSEstimator` on New York City Taxi Dataset. To run this example, install the following: `pip install --pre --upgrade bigdl-chronos[all]`.

```python
from bigdl.orca.automl import hp
from bigdl.chronos.data.repo_dataset import get_public_dataset
from bigdl.chronos.autots import AutoTSEstimator
from bigdl.orca import init_orca_context, stop_orca_context
from sklearn.preprocessing import StandardScaler

# initial orca context
init_orca_context(cluster_mode="local", cores=4, memory="8g")

# load dataset
tsdata_train, tsdata_val, tsdata_test = get_public_dataset(name='nyc_taxi')

# dataset preprocessing
stand = StandardScaler()
for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
    tsdata.gen_dt_feature().impute()\
          .scale(stand, fit=tsdata is tsdata_train)

# AutoTSEstimator initalization
autotsest = AutoTSEstimator(model="tcn",
                            past_seq_len=hp.randint(50, 200),
                            future_seq_len=10)

# AutoTSEstimator fitting
tsppl = autotsest.fit(tsdata_train,
                      validation_data=tsdata_val)

# Evaluation
autotsest_mse = tsppl.evaluate(tsdata_test)

# stop orca context
stop_orca_context()
```

### **5. Details**
_Chronos_ provides flexible components for forecasting, detection, simulation and other userful functionalities. You may review following pages to fully learn how to use Chronos to build various time series related applications.
- [Time Series Processing and Feature Engineering Overview](./data_processing_feature_engineering.html)
- [Time Series Forecasting Overview](./forecasting.html)
- [Time Series Anomaly Detection Overview](./anomaly_detection.html)
- [Generate Synthetic Sequential Data Overview](./simulation.html)
- [Useful Functionalities Overview](./useful_functionalities.html)
- [Chronos API Doc](../../PythonAPI/Chronos/index.html)

### **6. Examples and Demos**
- Quickstarts
    - [Use AutoTSEstimator for Time-Series Forecasting](https://bigdl.readthedocs.io/en/latest/doc/Chronos/QuickStart/chronos-autotsest-quickstart.html)
    - [Use TSDataset and Forecaster for Time-Series Forecasting](https://bigdl.readthedocs.io/en/latest/doc/Chronos/QuickStart/chronos-tsdataset-forecaster-quickstart.html)
    - [Use Anomaly Detector for Unsupervised Anomaly Detection](https://bigdl.readthedocs.io/en/latest/doc/Chronos/QuickStart/chronos-anomaly-detector.html)
- Examples
    - [Use AutoLSTM on nyc taxi dataset](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/example/auto_model/autolstm_nyc_taxi.py)
    - [Use AutoProphet on nyc taxi dataset](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/example/auto_model/autoprophet_nyc_taxi.py)
    - [High dimension time series forecasting with Chronos TCMFForecaster](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/example/tcmf/run_electricity.py)
    - [Use distributed training with Chronos Seq2SeqForecaster](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/example/distributed/distributed_training_network_traffic.py)
    - [Use ONNXRuntime to accelerate the inference of AutoTSEstimator](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/example/onnx/onnx_autotsestimator_nyc_taxi.py)
    - [Use ONNXRuntime to accelerate the inference of Seq2SeqForecaster](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/example/onnx/onnx_forecaster_network_traffic.py)
    - [Generate synthetic data with DPGANSimulator in a data-driven fashion](https://github.com/intel-analytics/BigDL/tree/branch-2.0/python/chronos/example/simulator)
- Use cases
    - [Unsupervised Anomaly Detection](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.ipynb)
    - [Unsupervised Anomaly Detection based on Forecasts](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.ipynb)
    - [Stock Price Prediction with LSTM](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/fsi/stock_prediction.ipynb)
    - [Stock Price Prediction with ProphetForecaster and AutoProphet](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/fsi/stock_prediction_prophet.ipynb)
    - [Network Traffic Forecasting with AutoTSEstimator](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting.ipynb)
    - [Network Traffic Forecasting (using multivariate time series data)](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/network_traffic/network_traffic_model_forecasting.ipynb)
    - [Network Traffic Forecasting (using multistep time series data)](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.ipynb)
    - [Network Traffic Forecasting with Customized Model](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model.ipynb)
