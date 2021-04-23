# Zouwu User Guide

### **1. Overview**
_Zouwu_ is an application framework for building large-scale time series analysis applications.

There are two ways to use _Zouwu_ for time series analysis:

- AutoML enabled pipelines (i.e. [AutoTS](#3-use-autots-pipeline-with-automl))
- Standalone [forecast pipeline](#4-use-standalone-forecaster-pipeline) without AutoML

### **2. Install**

_Zouwu_ depends on the Python libraries below:

```bash
python 3.6 or 3.7
pySpark
analytics-zoo
tensorflow>=1.15.0,<2.0.0
h5py==2.10.0
ray[tune]==1.2.0
psutil
aiohttp
setproctitle
pandas
scikit-learn>=0.20.0,<0.24.0
requests
```

You can easily install all the dependencies for _Zouwu_ as follows:

```bash
conda create -n my_env python=3.7
conda activate my_env
pip install --pre --upgrade analytics-zoo[automl]
```

---
### **3. Use AutoTS Pipeline (with AutoML)**

You can use the ```AutoTS``` package to to build a time series forecasting pipeline with AutoML.

The general workflow has two steps:

* Create a [AutoTSTrainer](#33-create-autotstrainer) and train; it will then return a [TSPipeline](#35-use-tspipeline).
* Use [TSPipeline](#35-use-tspipeline) to do prediction, evaluation, and incremental fitting.

View [AutoTS example](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/zouwu/use-case/network_traffic/network_traffic_autots_forecasting.ipynb) for more details.

#### **3.1 Initialize Orca Context**

AutoTS uses [RayOnSpark](./ray.md) to train (or `fit`) the time series pipeline, and needs to call `init_orca_context` with argument `init_ray_on_spark=True`. 

View [Orca Context](../Orca/Overview/orca-context.md) for more details. 

* Local mode

```python
from zoo.orca import init_orca_context, stop_orca_context
init_orca_context(cluster_mode="local", cores=4, memory='2g', init_ray_on_spark=True)
```

* YARN client mode

```python
from zoo.orca import init_orca_context, stop_orca_context
init_orca_context(cluster_mode="yarn-client",
                  num_nodes=2, cores=2,
                  conda_name='my_env',
                  extra_memory_for_ray="10g",
                  object_store_memory='5g',
                  init_ray_on_spark=True)
```

#### **3.2 Prepare input data**

You should prepare the training dataset and the optional validation dataset. Both training and validation data need to be provided as *Pandas Dataframe*.  The dataframe should have at least two columns:
- The *datetime* column, which should have Pandas datetime format (you can use `pandas.to_datetime` to convert a string into a datetime format)
- The *target* column, which contains the data points at the associated timestamps; these data points will be used to predict future data points. 

You may have other input columns for each row as extra feature; so the final input data could look something like below.

```bash
datetime    target  extra_feature_1  extra_feature_2
2019-06-06  1.2     1                2
2019-06-07  2.30    2                1
```

#### **3.3 Create AutoTSTrainer**

You can create an `AutoTSTrainer` as follows (`dt_col` is the datetime, `target_col` is the target column, and `extra_features_col` is the extra features):

```python
from zoo.zouwu.autots.forecast import AutoTSTrainer

trainer = AutoTSTrainer(dt_col="datetime",
                        target_col="target",
                        horizon=1,
                        extra_features_col=["extra_feature_1","extra_feature_2"])
```

Refer to [AutoTSTrainer API Doc]() for more details.

#### **3.4 Train AutoTS pipeline**

You can then train on the input data using `AutoTSTrainer.fit` with AutoML as follows:

```python
ts_pipeline = trainer.fit(train_df, validation_df)
```

After training, it will return a TSPipeline, which includes not only the model, but also the data preprocessing/post processing steps. 

Appropriate hyperparameters are automatically selected for the models and data processing steps in the pipeline during the fit process, and you may use built-in [visualization tool](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/AutoML/visualization.md) to inspect the training results after training stopped.

#### **3.5 Use TSPipeline**

Use `TSPipeline.predict|evaluate|fit` for prediction, evaluation or (incremental) fitting. (Note that incremental fitting on TSPipeline just update the model weights the standard way, which does not involve AutoML).

```python
#predict
ts_pipeline.predict(test_df)
#evaluate
ts_pipeline.evalute(val_df)
#incremental fitting
ts_pipeline.fit(new_train_df, new_val_df, epochs=10)
```

Use ```TSPipeline.save|load``` to load or save.

```python
from zoo.zouwu.autots.forecast import TSPipeline
loaded_ppl = TSPipeline.load(file)
# ... do sth. e.g. incremental fitting
loaded_ppl.save(another_file)
```

**Note**:  `init_orca_context` is not needed if you just use the trained TSPipeline for inference, evaluation or incremental fitting.

---
### **4. Use Standalone Forecaster Pipeline**

Zouwu also provides a set of standalone time series forecaster, which are based on deep learning models (without AutoML support), including

* TCNForecaster
* LSTMForecaster
* MTNetForecaster
* TCMFForecaster

View [Network Traffic Prediction](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/zouwu/use-case/network_traffic/network_traffic_model_forecasting.ipynb) and [Datacenter AIOps](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/zouwu/use-case/AIOps) notebooks for some examples.

#### **4.1 Initialize Orca Context**

First, call `init_orca_context` as follows 

* Local mode

```python
from zoo.orca import init_orca_context, stop_orca_context
init_orca_context(cluster_mode="local", cores=4, memory='2g')
```

* YARN client mode

```python
from zoo.orca import init_orca_context, stop_orca_context
init_orca_context(cluster_mode="yarn-client",
                  num_nodes=2, cores=2,
                  conda_name='my_env')
```

View [Orca Context](../Orca/Overview/orca-context.md) for more details. 

#### **4.2 Create a Forecaster**

Next, create an appropriate Forecaster to fit, evaluate or predict on the input data.

##### **4.2.1 LSTMForecaster**

LSTMForecaster wraps a vanilla LSTM model, and is suitable for univariate time series forecasting.

View Network Traffic Prediction [notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/zouwu/use-case/network_traffic/network_traffic_model_forecasting.ipynb) and [LSTMForecaster API Doc]() for more details.

##### **4.2.2 MTNetForecaster**

MTNetForecaster wraps a MTNet model. The model architecture mostly follows the [MTNet paper](https://arxiv.org/abs/1809.02105) with slight modifications, and is suitable for multivariate time series forecasting.

View Network Traffic Prediction [notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/zouwu/use-case/network_traffic/network_traffic_model_forecasting.ipynb) and [MTNetForecaster API Doc]() for more details.

##### **4.2.3 TCMFForecaster**

TCMFForecaster wraps a model architecture that follows implementation of the paper [DeepGLO paper](https://arxiv.org/abs/1905.03806) with slight modifications. It is especially suitable for extremely high dimensional (up-to millions) multivariate time series forecasting.

View High-dimensional Electricity Data Forecasting [example](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/zouwu/examples/tcmf/run_electricity.py) and [TCMFForecaster API Doc]() for more details.
