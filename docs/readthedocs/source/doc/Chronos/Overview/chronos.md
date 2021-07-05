# Chronos User Guide

### **1. Overview**
_Chronos_ is an application framework for building large-scale time series analysis applications.

You can use _Chronos_ to do:

- Time Series Forecasting (using [Standalone Forecasters](#use-standalone-forecaster-pipeline) or [AutoTS](#use-autots-pipeline-with-automl) (AutoML enabled pipelines))
- Anomaly Detection (using [Anomaly Detectors](#anomaly-detection))
- Data preprocessing and feature generation (using [TSDataset](#data-processing-and-features))

### **2. Install**

Install analytics-zoo with target `[automl]` to install the additional dependencies for _Chronos_. 

```bash
conda create -n my_env python=3.7
conda activate my_env
pip install --pre --upgrade analytics-zoo[automl]
```

### **3 Initialization**

_Chronos_ uses [Orca](../../Orca/Overview/orca.md) to enable distributed training and AutoML capabilities. Init orca as below. View [Orca Context](../../Orca/Overview/orca-context.md) for more details. Note that argument `init_ray_on_spark` must be `True` for _Chronos_. 

```python
if args.cluster_mode == "local":
    init_orca_context(cluster_mode="local", cores=4, init_ray_on_spark=True) # run in local mode
elif args.cluster_mode == "k8s":
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=2, init_ray_on_spark=True) # run on K8s cluster
elif args.cluster_mode == "yarn":
    init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2, init_ray_on_spark=True) # run on Hadoop YARN cluster
```
View [Quick Start](../QuickStart/chronos-autots-quickstart.md) for a more detailed example. 

---
### **4 Forecasting** 

Time Series Forecasting uses the history to predict the future. There're two ways to do forecasting:

- Use AutoTS pipeline
- Use Standalone Forecaster pipeline

#### **4.1 Use AutoTS Pipeline (with AutoML)**

You can use the ```AutoTS``` package to to build a time series forecasting pipeline with AutoML.

The general workflow has two steps:

* Create a [AutoTSTrainer](../../PythonAPI/Chronos/autots.html#zoo.chronos.autots.forecast.AutoTSTrainer) and train; it will then return a [TSPipeline](../../PythonAPI/Chronos/autots.html#zoo.chronos.autots.forecast.TSPipeline).
* Use [TSPipeline](../../PythonAPI/Chronos/autots.html#zoo.chronos.autots.forecast.TSPipeline) to do prediction, evaluation, and incremental fitting.

View [AutoTS notebook example](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_autots_forecasting.ipynb) for more details.


##### **4.1.1 Prepare input data**

You should prepare the training dataset and the optional validation dataset. Both training and validation data need to be provided as *Pandas Dataframe*.  The dataframe should have at least two columns:
- The *datetime* column, which should have Pandas datetime format (you can use `pandas.to_datetime` to convert a string into a datetime format)
- The *target* column, which contains the data points at the associated timestamps; these data points will be used to predict future data points. 

You may have other input columns for each row as extra feature; so the final input data could look something like below.

```bash
datetime    target  extra_feature_1  extra_feature_2
2019-06-06  1.2     1                2
2019-06-07  2.30    2                1
```

##### **4.1.2 Create AutoTSTrainer**

You can create an `AutoTSTrainer` as follows (`dt_col` is the datetime, `target_col` is the target column, and `extra_features_col` is the extra features):

```python
from zoo.chronos.autots.forecast import AutoTSTrainer

trainer = AutoTSTrainer(dt_col="datetime", target_col="target", horizon=1, extra_features_col=["extra_feature_1","extra_feature_2"])
```

View [AutoTSTrainer API Doc](../../PythonAPI/Chronos/autots.html#zoo.chronos.autots.forecast.AutoTSTrainer) for more details.

##### **4.1.3 Train AutoTS pipeline**

You can then train on the input data using `AutoTSTrainer.fit` with AutoML as follows:

```python
ts_pipeline = trainer.fit(train_df, validation_df, recipe=SmokeRecipe())
```

`recipe` configures the search space for auto tuning. View [Recipe API docs](../../PythonAPI/Chronos/autots.html#chronos-config-recipe) for available recipes. 
After training, it will return a [TSPipeline](../../PythonAPI/Chronos/autots.html#zoo.chronos.autots.forecast.TSPipeline), which includes not only the model, but also the data preprocessing/post processing steps. 

Appropriate hyperparameters are automatically selected for the models and data processing steps in the pipeline during the fit process, and you may use built-in [visualization tool](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/AutoML/visualization.md) to inspect the training results after training stopped.


##### **4.1.4 Use TSPipeline**

Use `TSPipeline.predict|evaluate|fit` for prediction, evaluation or (incremental) fitting. **Note**: incremental fitting on TSPipeline just update the model weights the standard way, which does not involve AutoML.

```python
ts_pipeline.predict(test_df)
ts_pipeline.evalute(val_df)
ts_pipeline.fit(new_train_df, new_val_df, epochs=10)
```

Use ```TSPipeline.save|load``` to load or save.

```python
from zoo.chronos.autots.forecast import TSPipeline
loaded_ppl = TSPipeline.load(file)
loaded_ppl.save(another_file)
```

View [TSPipeline API Doc](../../PythonAPI/Chronos/autots.html#zoo.chronos.autots.forecast.TSPipeline) for more details.

**Note**:  `init_orca_context` is not needed if you just use the trained TSPipeline for inference, evaluation or incremental fitting.

---
#### **4.2 Use Standalone Forecaster Pipeline**

_Chronos_ provides a set of standalone time series forecasters without AutoML support, including deep learning models as well as traditional statistical models.

View some examples notebooks for [Network Traffic Prediction](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/network_traffic/) 

The common process of using a Forecaster looks like below. 
```python
f = Forecaster()
f.fit(...)
f.predict(...)
```
Refer to API docs of each Forecaster for detailed usage instructions and examples.

###### **4.2.1 LSTMForecaster**

LSTMForecaster wraps a vanilla LSTM model, and is suitable for univariate time series forecasting.

View Network Traffic Prediction [notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_model_forecasting.ipynb) and [LSTMForecaster API Doc](../../PythonAPI/Chronos/forecasters.html#chronos-model-forecast-lstm-forecaster) for more details.

###### **4.2.2 Seq2SeqForecaster**

Seq2SeqForecaster wraps a sequence to sequence model based on LSTM, and is suitable for multivariant & multistep time series forecasting.

View [Seq2SeqForecaster API Doc](../../PythonAPI/Chronos/forecasters.html#chronos-model-forecast-seq2seq-forecaster) for more details.

###### **4.2.3 TCNForecaster**

Temporal Convolutional Networks (TCN) is a neural network that use convolutional architecture rather than recurrent networks. It supports multi-step and multi-variant cases. Causal Convolutions enables large scale parallel computing which makes TCN has less inference time than RNN based model such as LSTM.

View Network Traffic multivariate multistep Prediction [notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.ipynb) and [TCNForecaster API Doc](../../PythonAPI/Chronos/forecasters.html#chronos-model-forecast-tcn-forecaster) for more details.

###### **4.2.4 MTNetForecaster**

MTNetForecaster wraps a MTNet model. The model architecture mostly follows the [MTNet paper](https://arxiv.org/abs/1809.02105) with slight modifications, and is suitable for multivariate time series forecasting.

View Network Traffic Prediction [notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_model_forecasting.ipynb) and [MTNetForecaster API Doc](../../PythonAPI/Chronos/forecasters.html#chronos-model-forecast-mtnet-forecaster) for more details.

###### **4.2.5 TCMFForecaster**

TCMFForecaster wraps a model architecture that follows implementation of the paper [DeepGLO paper](https://arxiv.org/abs/1905.03806) with slight modifications. It is especially suitable for extremely high dimensional (up-to millions) multivariate time series forecasting.

View High-dimensional Electricity Data Forecasting [example](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/examples/tcmf/run_electricity.py) and [TCMFForecaster API Doc](../../PythonAPI/Chronos/forecasters.html#chronos-model-forecast-tcmf-forecaster) for more details.

###### **4.2.6 ARIMAForecaster**

ARIMAForecaster wraps a ARIMA model and is suitable for univariate time series forecasting. It works best with data that show evidence of non-stationarity in the sense of mean (and an initial differencing step (corresponding to the "I, integrated" part of the model) can be applied one or more times to eliminate the non-stationarity of the mean function.

View [ARIMAForecaster API Doc](../../PythonAPI/Chronos/forecasters.html#chronos-model-forecast-arima-forecaster) for more details.

###### **4.2.7 ProphetForecaster**

ProphetForecaster wraps the Prophet model ([site](https://github.com/facebook/prophet)) which is an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects and is suitable for univariate time series forecasting. It works best with time series that have strong seasonal effects and several seasons of historical data and is robust to missing data and shifts in the trend, and typically handles outliers well.

View Stock Prediction [notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/fsi/stock_prediction_prophet.ipynb) and [ProphetForecaster API Doc](../../PythonAPI/Chronos/forecasters.html#chronos-model-forecast-prophet-forecaster) for more details.

### **5 Anomaly Detection**

Anomaly Detection detects abnormal samples in a given time series. _Chronos_ provides a set of unsupervised anomaly detectors. 

View some examples notebooks for [Datacenter AIOps](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/chronos/use-case/AIOps).

#### **5.1 ThresholdDetector**

ThresholdDetector detects anomaly based on threshold. It can be used to detect anomaly on a given time series ([notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.ipynb)), or used together with Forecasters (#forecasting) to detect anomaly on new coming samples ([notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.ipynb)). 

View [ThresholdDetector API Doc](../../PythonAPI/Chronos/anomaly_detectors.html#chronos-model-anomaly-th-detector) for more details.


#### **5.2 AEDetector**

AEDetector detects anomaly based on the reconstruction error of an autoencoder network. 

View anomaly detection [notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.ipynb) and [AEDetector API Doc](../../PythonAPI/Chronos/anomaly_detectors.html#chronos-model-anomaly-ae-detector) for more details.

#### **5.3 DBScanDetector**

DBScanDetector uses DBSCAN clustering algortihm for anomaly detection. 

View anomaly detection [notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.ipynb) and [DBScanDetector API Doc](../../PythonAPI/Chronos/anomaly_detectors.html#chronos-model-anomaly-dbscan-detector) for more details.

### **6 Data Processing and Features**

_Chronos_ provides TSDataset for time series data processing and feature engineering. 

View [TSDataset API Doc](../../PythonAPI/Chronos/tsdataset.html#) for more details. 
