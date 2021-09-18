# Auto model examples on nyc_taxi dataset
This example collection will demonstrate Chronos auto models (i.e. autotcn & autoprophet) perform automatic time series forecasting on nyc_taxi dataset. The auto model will search the best hyperparameters automatically.

## Prepare the environment
We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:
```
conda create -n zoo python=3.7 # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install --pre --upgrade analytics-zoo[automl]
```

## Prepare data
We are using the nyc taxi provided by NAB, from 2014-07-01 to 2015-01-31 taxi fare information For more details, you can download it [here](https://raw.githubusercontent.com/numenta/NAB/v1.0/data/realKnownCause/nyc_taxi.csv) and view the detailed information [here](https://github.com/numenta/NAB/tree/master/data).


## Run the example
For autolstm example
```bash
python autolstm_nyc_taxi.py
```
For autoprophet example
```bash
python autoprophet_nyc_taxi.py
```

## Options
* `--cluster_mode` The mode for the Spark cluster. local or yarn. Default to be `local`. You can refer to OrcaContext documents [here](https://analytics-zoo.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) for details.
* `--epoch` Max number of epochs to train in each trial. Default to be 1.
* `--cpus_per_trail` Number of cpus for each trial. Default to be 2.
* `--n_sampling` Number of times to sample from the search_space. Default to be 1.