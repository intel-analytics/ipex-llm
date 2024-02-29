# Use Chronos forecasters in a distributed fashion
LSTM, TCN, NBeats and Seq2seq users can easily train their forecasters in a distributed fashion to handle extra large dataset and speed up the process (training and data processing) by utilizing a cluster or pseudo-distribution on a single node. The functionality is powered by Project Orca.

## Prepare the environment
We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster.
Please refer to [Chronos Install Guide](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/install.html)

## Prepare data
Users may utilize data from different source (local file, file on HDFS, spark dataframe, etc.)
### Local data - through Pandas
we use the publicly available `network traffic` data repository maintained by the [WIDE project](http://mawi.wide.ad.jp/mawi/) and in particular, the network traffic traces aggregated every 2 hours (i.e. AverageRate in Mbps/Gbps and Total Bytes) in year 2018 and 2019 at the transit link of WIDE to the upstream ISP ([dataset link](http://mawi.wide.ad.jp/~agurim/dataset/))

`get_public_dataset` automatically download the specified data set and return the tsdata that can be used directly after preprocessing.
```python
# Just specify the name, (e.g. network_traffic)
name = 'network_traffic'
tsdata_train, _, tsdata_test = get_public_dataset(name)
minmax = MinMaxScaler()
for tsdata in [tsdata_train, tsdata_test]:
    tsdata.gen_dt_feature(one_hot_features=["HOUR", "WEEK"])\
          .impute("last")\
          .scale(minmax, fit=tsdata is tsdata_train)\
          .roll(lookback=100, horizon=10)
data_train = tsdata_train.to_numpy()
```
### Distributed data - through Spark
We also support our users to directly fetch and process their data natively under spark.
Please find detailed information under `sparkdf_training_network_traffic.py`.
```python
sc = OrcaContext.get_spark_context()
spark = OrcaContext.get_spark_session()
df = spark.read.format("csv")\
                .option("inferSchema", "true")\
                .option("header", "true")\
                .load(dataset_path)
tsdata_train, _, tsdata_test = XShardsTSDataset.from_sparkdf(df, dt_col="timestamp",
                                            target_col=["value"],
                                            with_split=True,
                                            val_ratio=0,
                                            test_ratio=0.1)
for tsdata in [tsdata_train, tsdata_test]:
    tsdata.roll(lookback=100, horizon=10)
data_train = tsdata_train.to_xshards()
```


## Initialize forecaster and fit
Initialize a forecaster and set `distributed=True` and optionally `workers_per_node`.
```python
forecaster = Seq2SeqForecaster(past_seq_len=100,
                               future_seq_len=10,
                               input_feature_num=x_train.shape[-1],
                               output_feature_num=2,
                               metrics=['mse'],
                               distributed=True,
                               workers_per_node=args.workers_per_node,
                               seed=0)

forecaster.fit(data_train, epochs=args.epochs)
```

## Evaluate
Use the same API as non-distributed version forecaster for evaluation/prediction. `Evaluator` is only valid for pandas backend. Users may use `forecaster.evaluate` to evaluate if using spark as backend for data processing.
```python
rmse, smape = [Evaluator.evaluate(m, y_true=unscale_y_test,
                                  y_pred=unscale_yhat,
                                  multioutput='raw_values') for m in ['rmse', 'smape']]
print(f'rmse is: {np.mean(rmse)}')
print(f'smape is: {np.mean(smape):.4f}')
```

## Options
Users may find detailed information about options through:
```python
python sparkdf_training_network_traffic.py --help
python distributed_training_network_traffic.py --help
```