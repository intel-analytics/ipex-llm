# Use Chronos forecasters in a distributed fashion
LSTM, TCN and Seq2seq users can easily train their forecasters in a distributed fashion to handle extra large dataset and speed up the process (especially the training) by utilizing a cluster or pseudo-distribution on a single node. The functionality is powered by Project Orca.

## Prepare the environment
We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:
```bash
conda create -n my_env python=3.7 # "my_env" is conda environment name, you can use any name you like.
conda activate my_env
pip install --pre --upgrade bigdl-chronos[all]
```
Please refer to [Chronos Install Guide](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html#install)

## Prepare data
we use the publicly available `network traffic` data repository maintained by the [WIDE project](http://mawi.wide.ad.jp/mawi/) and in particular, the network traffic traces aggregated every 2 hours (i.e. AverageRate in Mbps/Gbps and Total Bytes) in year 2018 and 2019 at the transit link of WIDE to the upstream ISP ([dataset link](http://mawi.wide.ad.jp/~agurim/dataset/))

`get_public_dataset` automatically download the specified data set and return the tsdata that can be used directly after preprocessing.
```python
# Just specify the name and path, (e.g. network_traffic)
name = 'network_traffic'
path = '~/.chronos/dataset/'
tsdata_train, _, tsdata_test = get_public_dataset(name, path,
                                                  redownload=False,
                                                  with_split=True,
                                                  test_ratio=0.1)
minmax = MinMaxScaler()
for tsdata in [tsdata_train, tsdata_test]:
    tsdata.gen_dt_feature(one_hot_features=["HOUR", "WEEK"])\
          .impute("last")\
          .scale(minmax, fit=tsdata is tsdata_train)\
          .roll(lookback=100, horizon=10)
```

## Initialize forecaster and fit
Initialize a forecaster and set `distributed=True` and optionally `workers_per_node`.
```python
x_train, y_train = tsdata_train.to_numpy()
forecaster = Seq2SeqForecaster(past_seq_len=100,
                               future_seq_len=10,
                               input_feature_num=x_train.shape[-1],
                               output_feature_num=2,
                               metrics=['mse'],
                               distributed=True,
                               workers_per_node=args.workers_per_node,
                               seed=0)

forecaster.fit((x_train, y_train), epochs=args.epochs,
               batch_size=512//(1 if not forecaster.distributed else args.workers_per_node))
```

## Evaluate
Use the same API as non-distributed version forecaster for evalution/prediction.
```python
rmse, smape = [Evaluator.evaluate(m, y_true=unscale_y_test,
                                  y_pred=unscale_yhat,
                                  multioutput='raw_values') for m in ['rmse', 'smape']]
print(f'rmse is: {np.mean(rmse)}')
print(f'smape is: {np.mean(smape):.4f}')
```

## Options
* `--cluster_mode` The mode for the Spark cluster. local or yarn. Default to be `local`. You can refer to OrcaContext documents [here](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) for details.
* `--memory` The memory you want to use on each node. You can change it depending on your own cluster setting. Default to be 32g.
* `--cores` The number of cpu cores you want to use on each node. You can change it depending on your own cluster setting. Default to be 4.
* `--epochs` Max number of epochs to train in each trial. Default to be 2.
* `--workers_per_node` the number of worker you want to use.The value defaults to 1. The param is only effective when distributed is set to True.
* `--num_workers` The number of workers to be used in the cluster. You can change it depending on your own cluster setting. Default to be 1.