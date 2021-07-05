# AutoLSTM examples on nyc_taxi dataset
This example will demonstrate that Auto LSTM performs automatic time series forecasting on nyc_taxi. Auto Lstm will return the best hyperparameter set within the specified hyperparameter range.

## Prepare the environment
We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:
```
conda create -n zoo python=3.7 # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install analytics-zoo
pip install torch==1.8.1 ray[tune]==1.2.0 scikit-learn
```

## Prepare data
We are using the nyc taxi provided by NAB, from 2014-07-01 to 2015-01-31 taxi fare information For more details, please refer to [here](https://raw.githubusercontent.com/numenta/NAB/v1.0/data/realKnownCause/nyc_taxi.csv)


## Run on local after pip install
```
python test_auto_lstm.py
```

## Run on yarn cluster for yarn-client mode after pip install 
```
python test_auto_lstm.py --cluster_model yarn
```

## Options
* `--cluster_mode` The mode for the Spark cluster. local or yarn. Default to be `local`. You can refer to OrcaContext documents [here](https://analytics-zoo.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) for details.
* `--epoch` Max number of epochs to train in each trial. Default to be 1.
* `--cpus_per_trail` Number of cpus for each trial. Default to be 2.
* `--n_sampling` Number of times to sample from the search_space. Default to be 1.