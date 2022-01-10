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
