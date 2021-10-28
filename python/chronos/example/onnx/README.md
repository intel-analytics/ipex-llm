# Use ONNX to speed up inferencing
This example will demonstrate how to use ONNX to speed up the inferencing(prediction/evalution) on forecasters and `AutoTSEstimator`. 
In this example, onnx speed up the inferencing for ~4X.

## Prepare the environment
We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:
```bash
conda create -n my_env python=3.7 # "my_env" is conda environment name, you can use any name you like.
conda activate my_env
pip install --pre --upgrade bigdl-chronos[all]
```

## Prepare data
**autotsest**: We are using the `nyc taxi` provided by NAB, from 2014-07-01 to 2015-01-31 taxi fare information For more details, please refer to [here](https://raw.githubusercontent.com/numenta/NAB/v1.0/data/realKnownCause/nyc_taxi.csv)

**forecaster**: For demonstration, we use the publicly available `network traffic` data repository maintained by the [WIDE project](http://mawi.wide.ad.jp/mawi/) and in particular, the network traffic traces aggregated every 2 hours (i.e. AverageRate in Mbps/Gbps and Total Bytes) in year 2018 and 2019 at the transit link of WIDE to the upstream ISP ([dataset link](http://mawi.wide.ad.jp/~agurim/dataset/))

First, `get_public_dataset` automatically download the specified data set and return the tsdata that can be used directly after preprocessing.
```python
# Just specify the name and path, (e.g. network_traffic)
name = 'network_traffic'
path = '~/.chronos/dataset/'
tsdata_train, _, tsdata_test = get_public_dataset(name, path, with_split=True, test_ratio=0.1)
minmax = MinMaxScaler()
for tsdata in [tsdata_train, tsdata_test]:
    tsdata.gen_dt_feature(one_hot_features=["HOUR", "WEEK"])\
          .impute("last")\
          .scale(minmax, fit=tsdata is tsdata_train)\
          .roll(lookback=100, horizon=10)
```

## Fit on forecaster/AutoTSEstimator
Create and fit on the forecaster/AutoTSEstimator. Please refer to [API doc](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/index.html) for detail.

## Inference with onnx
All methods involve inferencing supports onnx as backend. You can call `predict_with_onnx` or `evaluate_with_onnx` to use it.
```python
# forecaster
x_test, y_test = tsdata_train.to_numpy()
forecaster.predict_with_onnx(x_test)
forecaster.evaluate_with_onnx((x_test, y_test))

# TSpipeline
tspipeline.predict_with_onnx(tsdata_test)
tspipeline.evaluate_with_onnx(tsdata_test)
```

## Result
ONNX will not affect the result of evaluate, and will speed up predict.
```python
mse, smape = forecaster.evaluate((x_test,y_test))
# evaluate mse is: 0.0014
# evaluate smape is: 9.6629
mse, smape = forecaster.evaluate_with_onnx((x_test,y_test))
# evaluate_onnx mse is: 0.0014
# evaluate_onnx smape is: 9.6629

forecaster.predict(x_test)
# inference time is: 0.136s
forecaster.predict_with_onnx(x_test)
# inference(onnx) time is: 0.030s 
```

## Options
* `--epochs` Max number of epochs to train in each trial. Default to be 2.
* `--n_sampling` Number of times to sample from the search_space. Default to be 1.
* `--cpus_per_trail` Number of cpus for each trial. Default to be 2.
* `--memory` The memory you want to use on each node. Default to be 10g.
* `--cluster_mode` The mode for the Spark cluster. local or yarn. Default to be `local`. You can refer to OrcaContext documents [here](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) for details.
* `--cores` "The number of cpu cores you want to use on each node. Default to be 4.
* `--num_workers` The number of workers to be used in the cluster. You can change it depending on your own cluster setting. Default to be 2.
