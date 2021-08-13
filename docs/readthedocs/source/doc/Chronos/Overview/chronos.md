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

_Chronos_ uses [Orca](../../Orca/Overview/orca.md) to enable distributed training and AutoML capabilities. Init orca as below when you want to:

1. Use the distributed mode of a standalone forecaster.
2. Use automl to distributedly tuning your model.

View [Orca Context](../../Orca/Overview/orca-context.md) for more details. Note that argument `init_ray_on_spark` must be `True` for _Chronos_. 

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

#### **4.1 Time Series Forecasting Concepts**
Time series forecasting is one of the most popular tasks on time series data. **In short, forecasing aims at predicting the future by using the knowledge you can learn from the history.**

##### **4.1.1 Traditional Statistical(TS) Style**
Traditionally, Time series forecasting problem was formulated with rich mathematical fundamentals and statistical models. Typically, one model can only handle one time series and fit on the whole time series before the last observed timestamp and predict the next few steps. Training(fit) is needed every time you change the last observed timestamp.

![](../Image/forecast-TS.png)

##### **4.1.2 Regular Regression(RR) Style**
Recent years, common deep learning architectures (e.g. RNN, CNN, Transformer, etc.) are being successfully applied to forecasting problem. Forecasting is transformed to a supervised learning regression problem in this style. A model can predict several time series. Typically, a sampling process based on sliding-window is needed, some terminology is explained as following:

- `lookback` / `past_seq_len`: the length of historical data along time. This number is tunable.
- `horizon` / `future_seq_len`: the length of predicted data along time. This number is depended on the task definition. If this value larger than 1, then the forecasting task is *Multi-Step*.
- `input_feature_num`: The number of variables the model can observe. This number is tunable since we can select a subset of extra feature to use.
- `output_feature_num`: The number of variables the model to predict. This number is depended on the task definition. If this value larger than 1, then the forecasting task is *Multi-Variate*.

![](../Image/forecast-RR.png)

_Chronos_ provides both deep learning/machine learning models and traditional statistical models for forecasting. For details, please refer to [here](#supported_forecasting_model).

There're two ways to do forecasting:
- Use AutoTS pipeline
- Use Standalone Forecaster pipeline

#### **4.2 Use AutoTS Pipeline (with AutoML)**

You can use the ```AutoTS``` package to to build a time series forecasting pipeline with AutoML.

The general workflow has two steps:

* Create a [AutoTSTrainer](../../PythonAPI/Chronos/autots.html#zoo.chronos.autots.forecast.AutoTSTrainer) and train; it will then return a [TSPipeline](../../PythonAPI/Chronos/autots.html#zoo.chronos.autots.forecast.TSPipeline).
* Use [TSPipeline](../../PythonAPI/Chronos/autots.html#zoo.chronos.autots.forecast.TSPipeline) to do prediction, evaluation, and incremental fitting.

View [AutoTS notebook example](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_autots_forecasting.ipynb) for more details.


##### **4.2.1 Prepare input data**

You should prepare the training dataset and the optional validation dataset. Both training and validation data need to be provided as *Pandas Dataframe*.  The dataframe should have at least two columns:
- The *datetime* column, which should have Pandas datetime format (you can use `pandas.to_datetime` to convert a string into a datetime format)
- The *target* column, which contains the data points at the associated timestamps; these data points will be used to predict future data points. 

You may have other input columns for each row as extra feature; so the final input data could look something like below.

```bash
datetime    target  extra_feature_1  extra_feature_2
2019-06-06  1.2     1                2
2019-06-07  2.30    2                1
```

##### **4.2.2 Create AutoTSTrainer**

You can create an `AutoTSTrainer` as follows (`dt_col` is the datetime, `target_col` is the target column, and `extra_features_col` is the extra features):

```python
from zoo.chronos.autots.forecast import AutoTSTrainer

trainer = AutoTSTrainer(dt_col="datetime", target_col="target", horizon=1, extra_features_col=["extra_feature_1","extra_feature_2"])
```

View [AutoTSTrainer API Doc](../../PythonAPI/Chronos/autots.html#zoo.chronos.autots.forecast.AutoTSTrainer) for more details.

##### **4.2.3 Train AutoTS pipeline**

You can then train on the input data using `AutoTSTrainer.fit` with AutoML as follows:

```python
ts_pipeline = trainer.fit(train_df, validation_df, recipe=SmokeRecipe())
```

`recipe` configures the search space for auto tuning. View [Recipe API docs](../../PythonAPI/Chronos/autots.html#chronos-config-recipe) for available recipes. 
After training, it will return a [TSPipeline](../../PythonAPI/Chronos/autots.html#zoo.chronos.autots.forecast.TSPipeline), which includes not only the model, but also the data preprocessing/post processing steps. 

Appropriate hyperparameters are automatically selected for the models and data processing steps in the pipeline during the fit process, and you may use built-in [visualization tool](https://analytics-zoo.github.io/master/#ProgrammingGuide/AutoML/visualization/) (This link lead to our old document, please head back after reading the visualization page.) to inspect the training results after training stopped. 


##### **4.2.4 Use TSPipeline**

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
<span id="supported_forecasting_model"></span>
#### **4.3 Use Standalone Forecaster Pipeline**

_Chronos_ provides a set of standalone time series forecasters without AutoML support, including deep learning models as well as traditional statistical models.


| Forecaster        | Style | Multi-Variate | Multi-Step | Distributed |
| ----------------- | ----- | ------------- | ---------- | ----------- |
| [LSTMForecaster](#LSTMForecaster)    | RR    | ✅             | ❌          | ✅           |
| [Seq2SeqForecaster](#Seq2SeqForecaster)     | RR    | ✅             | ✅          | ✅           |
| [TCNForecaster](#TCNForecaster) | RR    | ✅             | ✅          | ✅           |
| [MTNetForecaster](#MTNetForecaster)   | RR    | ✅             | ✳️          | ✅           |
| [TCMFForecaster](#TCMFForecaster)    | TS    | ✅             | ✅          | ✳️           |
| [ProphetForecaster](#ProphetForecaster) | TS    | ❌             | ✅          | ❌           |
| [ARIMAForecaster](#ARIMAForecaster)   | TS    | ❌             | ✅          | ❌           |

View some examples notebooks for [Network Traffic Prediction](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/network_traffic/) 

The common process of using a Forecaster looks like below. 
```python
f = Forecaster(...)
f.fit(...)
f.predict(...)
```
View [Quick Start](../QuickStart/chronos-tsdataset-forecaster-quickstart.md) for a more detailed example. Refer to [API docs](../../PythonAPI/Chronos/forecasters.html) of each Forecaster for detailed usage instructions and examples.

<span id="LSTMForecaster"></span>
###### **4.3.1 LSTMForecaster**

LSTMForecaster wraps a vanilla LSTM model, and is suitable for univariate time series forecasting.

View Network Traffic Prediction [notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_model_forecasting.ipynb) and [LSTMForecaster API Doc](../../PythonAPI/Chronos/forecasters.html#chronos-model-forecast-lstm-forecaster) for more details.

<span id="Seq2SeqForecaster"></span>
###### **4.3.2 Seq2SeqForecaster**

Seq2SeqForecaster wraps a sequence to sequence model based on LSTM, and is suitable for multivariant & multistep time series forecasting.

View [Seq2SeqForecaster API Doc](../../PythonAPI/Chronos/forecasters.html#chronos-model-forecast-seq2seq-forecaster) for more details.

<span id="TCNForecaster"></span>
###### **4.3.3 TCNForecaster**

Temporal Convolutional Networks (TCN) is a neural network that use convolutional architecture rather than recurrent networks. It supports multi-step and multi-variant cases. Causal Convolutions enables large scale parallel computing which makes TCN has less inference time than RNN based model such as LSTM.

View Network Traffic multivariate multistep Prediction [notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.ipynb) and [TCNForecaster API Doc](../../PythonAPI/Chronos/forecasters.html#chronos-model-forecast-tcn-forecaster) for more details.

<span id="MTNetForecaster"></span>
###### **4.3.4 MTNetForecaster**

MTNetForecaster wraps a MTNet model. The model architecture mostly follows the [MTNet paper](https://arxiv.org/abs/1809.02105) with slight modifications, and is suitable for multivariate time series forecasting.

View Network Traffic Prediction [notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_model_forecasting.ipynb) and [MTNetForecaster API Doc](../../PythonAPI/Chronos/forecasters.html#chronos-model-forecast-mtnet-forecaster) for more details.

<span id="TCMFForecaster"></span>
###### **4.3.5 TCMFForecaster**

TCMFForecaster wraps a model architecture that follows implementation of the paper [DeepGLO paper](https://arxiv.org/abs/1905.03806) with slight modifications. It is especially suitable for extremely high dimensional (up-to millions) multivariate time series forecasting.

View High-dimensional Electricity Data Forecasting [example](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/examples/tcmf/run_electricity.py) and [TCMFForecaster API Doc](../../PythonAPI/Chronos/forecasters.html#chronos-model-forecast-tcmf-forecaster) for more details.

<span id="ARIMAForecaster"></span>
###### **4.3.6 ARIMAForecaster**

ARIMAForecaster wraps a ARIMA model and is suitable for univariate time series forecasting. It works best with data that show evidence of non-stationarity in the sense of mean (and an initial differencing step (corresponding to the "I, integrated" part of the model) can be applied one or more times to eliminate the non-stationarity of the mean function.

View [ARIMAForecaster API Doc](../../PythonAPI/Chronos/forecasters.html#chronos-model-forecast-arima-forecaster) for more details.

<span id="ProphetForecaster"></span>
###### **4.3.7 ProphetForecaster**

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

### **6 Data Processing and Feature Engineering**

Time series data is a special data formulation with its specific operations. _Chronos_ provides [`TSDataset`](../../PythonAPI/Chronos/tsdataset.html) as a time series dataset abstract for data processing (e.g. impute, deduplicate, resample, scale/unscale, roll sampling) and auto feature engineering (e.g. datetime feature, aggregation feature). Cascade call is supported for most of the methods. [`TSDataset`](../../PythonAPI/Chronos/tsdataset.html) can be initialized from a pandas dataframe and be directly used in `AutoTSEstimator`. It can also be converted to a pandas dataframe or numpy ndarray for Forecasters and Anomaly Detectors.

[`TSDataset`](../../PythonAPI/Chronos/tsdataset.html) is designed for general time series processing while providing many specific operations for the convenience of different tasks(e.g. forecasting, anomaly detection).

#### **6.1 Basic concepts**
A time series can be interpreted as a sequence of real value whose order is timestamp. While a time series dataset can be a combination of one or a huge amount of time series. It may contain multiple time series since users may collect different time series in the same/different period of time (e.g. An AIops dataset may have CPU usage ratio and memory usage ratio data for two servers at a period of time. This dataset contains four time series). 

In [`TSDataset`](../../PythonAPI/Chronos/tsdataset.html), we provide **2** possible dimensions to construct a high dimension time series dataset (i.e. **feature dimension** and **id dimension**).
* feature dimension: Time series along this dimension might be independent or related. Though they may be related, they are assumed to have **different patterns and distributions** and collected on the **same period of time**. For example, the CPU usage ratio and Memory usage ratio for the same server at a period of time.
* id dimension: Time series along this dimension are assumed to have the **same patterns and distributions** and might by collected on the **same or different period of time**. For example, the CPU usage ratio for two servers at a period of time.

All the preprocessing operations will be done on each independent time series(i.e on both feature dimension and id dimension), while feature scaling will be only carried out on the feature dimension.

#### **6.2 Create a TSDataset**
Currently [`TSDataset`](../../PythonAPI/Chronos/tsdataset.html) only supports initializing from a pandas dataframe through [`TSDataset.from_pandas`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.from_pandas). A typical valid time series dataframe `df` is shown below.

You can initialize a [`TSDataset`](../../PythonAPI/Chronos/tsdataset.html) by simply:
```python
# Server id  Datetime         CPU usage   Mem usage
# 0          08:39 2021/7/9   93          24            
# 0          08:40 2021/7/9   91          24              
# 0          08:41 2021/7/9   93          25              
# 0          ...              ...         ...
# 1          08:39 2021/7/9   73          79            
# 1          08:40 2021/7/9   72          80              
# 1          08:41 2021/7/9   79          80              
# 1          ...              ...         ...
tsdata = TSDataset.from_pandas(df,
                               dt_col="Datetime",
                               id_col="Server id",
                               target_col=["CPU usage",
                                           "Mem usage"])
```
`target_col` is a list of all elements along feature dimension, while `id_col` is the identifier that distinguishes the id dimension. `dt_col` is the datetime column. For `extra_feature_col`(not shown in this case), you should list those features that you are not interested for your task (e.g. you will **not** perform forecasting or anomaly detection task on this col).

If you are building a prototype for your forecasting/anomaly detection task and you need to split you dataset to train/valid/test set, you can use `with_split` parameter.[`TSDataset`](../../PythonAPI/Chronos/tsdataset.html) supports split with ratio by `val_ratio` and `test_ratio`.
#### **6.3 Time series dataset preprocessing**
[`TSDataset`](../../PythonAPI/Chronos/tsdataset.html) now supports [`impute`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.impute), [`deduplicate`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.deduplicate) and [`resample`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.resample). You may fill the missing point by [`impute`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.impute) in different modes. You may remove the records that are totally the same by [`deduplicate`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.deduplicate). You may change the sample frequency by [`resample`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.resample). A typical cascade call for preprocessing is:
```python
tsdata.deduplicate().resample(interval="2s").impute()
```
#### **6.4 Feature scaling**
Scaling all features to one distribution is important, especially when we want to train a machine learning/deep learning system. [`TSDataset`](../../PythonAPI/Chronos/tsdataset.html) supports all the scalers in sklearn through [`scale`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.scale) and [`unscale`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.unscale) method. Since a scaler should not fit on the validation and test set, a typical call for scaling operations is:
```python
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
# scale
for tsdata in [tsdata_train, tsdata_valid, tsdata_test]:
    tsdata.scale(scaler, fit=tsdata is tsdata_train)
# unscale
for tsdata in [tsdata_train, tsdata_valid, tsdata_test]:
    tsdata.unscale()
```
[`unscale_numpy`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.unscale_numpy) is specially designed for forecasters. Users may unscale the output of a forecaster by this operation. A typical call is:
```python
x, y = tsdata_test.scale(scaler)\
                  .roll(lookback=..., horizon=...)\
                  .to_numpy()
yhat = forecaster.predict(x)
unscaled_yhat = tsdata_test.unscale_numpy(yhat)
unscaled_y = tsdata_test.unscale_numpy(y)
# calculate metric by unscaled_yhat and unscaled_y
```
#### **6.5 Feature generation**
Other than historical target data and other extra feature provided by users, some additional features can be generated automatically by [`TSDataset`](../../PythonAPI/Chronos/tsdataset.html). [`gen_dt_feature`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.gen_dt_feature) helps users to generate 10 datetime related features(e.g. MONTH, WEEKDAY, ...). [`gen_global_feature`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.gen_global_feature) and [`gen_rolling_feature`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.gen_rolling_feature) are powered by tsfresh to generate aggregated features (e.g. min, max, ...) for each time series or rolling windows respectively.
#### **6.6 Roll sampling and other transformation**
Roll sampling (or sliding window sampling) is useful when you want to train a supervised deep learning forecasting model. Please refer to the API doc [`roll`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.roll) for detailed behavior. A typical call of [`roll`](../../PythonAPI/Chronos/tsdataset.html#zoo.chronos.data.tsdataset.TSDataset.roll) is as following:
```python
# forecaster
x, y = tsdata.roll(lookback=..., horizon=...).to_numpy()
forecaster.fit(x, y)

# anomaly detector on "target" col
x = tsdata.to_pandas()["target"].to_numpy()
anomaly_detector.fit(x)
```
View [TSDataset API Doc](../../PythonAPI/Chronos/tsdataset.html#) for more details. 
