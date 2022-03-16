# Time Series Processing and Feature Engineering Overview

Time series data is a special data formulation with its specific operations. _Chronos_ provides [`TSDataset`](../../PythonAPI/Chronos/tsdataset.html) as a time series dataset abstract for data processing (e.g. impute, deduplicate, resample, scale/unscale, roll sampling) and auto feature engineering (e.g. datetime feature, aggregation feature). Cascade call is supported for most of the methods. [`TSDataset`](../../PythonAPI/Chronos/tsdataset.html) can be initialized from a pandas dataframe and be directly used in [`AutoTSEstimator`](../../PythonAPI/Chronos/autotsestimator.html#autotsestimator). It can also be converted to a pandas dataframe or numpy ndarray for Forecasters and Anomaly Detectors.

[`TSDataset`](../../PythonAPI/Chronos/tsdataset.html) is designed for general time series processing while providing many specific operations for the convenience of different tasks(e.g. forecasting, anomaly detection).

## **1. Basic concepts**
A time series can be interpreted as a sequence of real value whose order is timestamp. While a time series dataset can be a combination of one or a huge amount of time series. It may contain multiple time series since users may collect different time series in the same/different period of time (e.g. An AIops dataset may have CPU usage ratio and memory usage ratio data for two servers at a period of time. This dataset contains four time series). 

In [`TSDataset`](../../PythonAPI/Chronos/tsdataset.html), we provide **2** possible dimensions to construct a high dimension time series dataset (i.e. **feature dimension** and **id dimension**).
* feature dimension: Time series along this dimension might be independent or related. Though they may be related, they are assumed to have **different patterns and distributions** and collected on the **same period of time**. For example, the CPU usage ratio and Memory usage ratio for the same server at a period of time.
* id dimension: Time series along this dimension are assumed to have the **same patterns and distributions** and might by collected on the **same or different period of time**. For example, the CPU usage ratio for two servers at a period of time.

All the preprocessing operations will be done on each independent time series(i.e on both feature dimension and id dimension), while feature scaling will be only carried out on the feature dimension.

## **2. Create a TSDataset**
Currently [`TSDataset`](../../PythonAPI/Chronos/tsdataset.html) supports initializing from a pandas dataframe through [`TSDataset.from_pandas`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.from_pandas) or from a parquet file through [`TSDataset.from_parquet`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.from_parquet). A typical valid time series dataframe `df` is shown below.

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
## **3. Time series dataset preprocessing**
[`TSDataset`](../../PythonAPI/Chronos/tsdataset.html) now supports [`impute`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.impute), [`deduplicate`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.deduplicate) and [`resample`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.resample). You may fill the missing point by [`impute`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.impute) in different modes. You may remove the records that are totally the same by [`deduplicate`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.deduplicate). You may change the sample frequency by [`resample`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.resample). A typical cascade call for preprocessing is:
```python
tsdata.deduplicate().resample(interval="2s").impute()
```
## **4. Feature scaling**
Scaling all features to one distribution is important, especially when we want to train a machine learning/deep learning system. [`TSDataset`](../../PythonAPI/Chronos/tsdataset.html) supports all the scalers in sklearn through [`scale`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.scale) and [`unscale`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.unscale) method. Since a scaler should not fit on the validation and test set, a typical call for scaling operations is:
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
[`unscale_numpy`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.unscale_numpy) is specially designed for forecasters. Users may unscale the output of a forecaster by this operation. A typical call is:
```python
x, y = tsdata_test.scale(scaler)\
                  .roll(lookback=..., horizon=...)\
                  .to_numpy()
yhat = forecaster.predict(x)
unscaled_yhat = tsdata_test.unscale_numpy(yhat)
unscaled_y = tsdata_test.unscale_numpy(y)
# calculate metric by unscaled_yhat and unscaled_y
```
## **5. Feature generation**
Other than historical target data and other extra feature provided by users, some additional features can be generated automatically by [`TSDataset`](../../PythonAPI/Chronos/tsdataset.html). [`gen_dt_feature`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.gen_dt_feature) helps users to generate 10 datetime related features(e.g. MONTH, WEEKDAY, ...). [`gen_global_feature`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.gen_global_feature) and [`gen_rolling_feature`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.gen_rolling_feature) are powered by tsfresh to generate aggregated features (e.g. min, max, ...) for each time series or rolling windows respectively.
## **6. Sampling and exporting**
A time series dataset needs to be sampling and exporting as numpy ndarray/dataloader to be used in machine learning and deep learning models(e.g. forecasters, anomaly detectors, auto models, etc.).
```eval_rst
.. warning::
    You don't need to call any sampling or exporting methods introduced in this section when using `AutoTSEstimator`.
```
### **6.1 Roll sampling**
Roll sampling (or sliding window sampling) is useful when you want to train a RR type supervised deep learning forecasting model. It works as the [diagram](#RR-forecast-image) shows. Please refer to the API doc [`roll`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.roll) for detailed behavior. Users can simply export the sampling result as numpy ndarray by [`to_numpy`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.to_numpy) or pytorch dataloader [`to_torch_data_loader`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.to_torch_data_loader).

```eval_rst
.. note:: 
    **Difference between `roll` and `to_torch_data_loader`**:
    
    `.roll(...)` performs the rolling before RR forecasters/auto models training while `.to_torch_data_loader(roll=True, ...)` performs rolling during the training.
    
    It is fine to use either of them when you have a relatively small dataset (less than 1G). `.to_torch_data_loader(roll=True, ...)` is recommended when you have a large dataset (larger than 1G) to save memory usage.
```

```eval_rst
.. note:: 
    **Roll sampling format**:
    
    As decribed in RR style forecasting concept, the sampling result will have the following shape requirement.

    | x: (sample_num, lookback, input_feature_num)
    | y: (sample_num, horizon, output_feature_num)

    Please follow the same shape if you use customized data creator.
```

A typical call of [`roll`](../../PythonAPI/Chronos/tsdataset.html#bigdl.chronos.data.tsdataset.TSDataset.roll) is as following:
```python
# forecaster
x, y = tsdata.roll(lookback=..., horizon=...).to_numpy()
forecaster.fit((x, y))
```

### **6.2 Pandas Exporting**
Now we support pandas dataframe exporting through `to_pandas()` for users to carry out their own transformation. Here is an example of using only one time series for anomaly detection.
```python
# anomaly detector on "target" col
x = tsdata.to_pandas()["target"].to_numpy()
anomaly_detector.fit(x)
```
View [TSDataset API Doc](../../PythonAPI/Chronos/tsdataset.html#) for more details. 

## **7. Built-in Dataset**

Built-in Dataset supports the function of data downloading, preprocessing, and returning to the `TSDataset` object of the public data set.

|Dataset name|Task|Time Series Length|Number of Instances|Feature Number|Information Page|Download Link|
|---|---|---|---|---|---|---|
|network_traffic|forecasting|8760|1|2|[network_traffic](http://mawi.wide.ad.jp/~agurim/about.html)|[network_traffic](http://mawi.wide.ad.jp/~agurim/dataset/)|
|nyc_taxi|forecasting|10320|1|1|[nyc_taxi](https://github.com/numenta/NAB/blob/master/data/README.md)|[nyc_taxi](https://raw.githubusercontent.com/numenta/NAB/v1.0/data/realKnownCause/nyc_taxi.csv)|
|fsi|forecasting|1259|1|1|[fsi](https://github.com/CNuge/kaggle-code/tree/master/stock_data)|[fsi](https://github.com/CNuge/kaggle-code/raw/master/stock_data/individual_stocks_5yr.zip)|
|AIOps|anomaly_detect|61570|1|1|[AIOps](https://github.com/alibaba/clusterdata)|[AIOps](http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/machine_usage.tar.gz)|
|uci_electricity|forecasting|140256|370|1|[uci_electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)|[uci_electricity](https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip)|

Specify the `name`, the raw data file will be saved in the specified `path` (defaults to ~/.chronos/dataset). `redownload` can help you re-download the files you need.

When `with_split` is set to True, the length of the data set will be divided according to the specified `val_ratio` and `test_ratio`, and three `TSDataset` will be returned. `with_split` defaults to False, that is, only one `TSDataset` is returned.
About `TSDataset`, more details, please refer to [here](../../PythonAPI/Chronos/tsdataset.html).

```python
# load built-in dataset
from bigdl.chronos.data.repo_dataset import get_public_dataset
from sklearn.preprocessing import StandardScaler
tsdata_train, tsdata_val, \
    tsdata_test = get_public_dataset(name='nyc_taxi',
                                     with_split=True,
                                     val_ratio=0.1,
                                     test_ratio=0.1
                                     )
# carry out additional customized preprocessing on the dataset.
stand = StandardScaler()
for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
    tsdata.gen_dt_feature(one_hot_features=['HOUR'])\
          .impute()\
          .scale(stand, fit=tsdata is tsdata_train)
```
