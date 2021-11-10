# High dimension time series forecasting with Chronos TCMFForecaster 

This example demonstrates how to use BigDL Chronos TCMFForecaster to run distributed training 
and inference for high dimension time series forecasting task.


## Environment
We recommend you to use Anaconda to prepare the environment.
```bash
conda create -n my_env python=3.7 # "my_env" is conda environment name, you can use any name you like.
conda activate my_env
pip install --pre --upgrade bigdl-chronos[all]
```
Please refer to [Chronos Install Guide](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html#install)

## Prepare data
The example use the public real-world electricity datasets. You can download by running [download datasets script](https://github.com/rajatsen91/deepglo/blob/master/datasets/download-data.sh). Note that we only need electricity.npy.

If you only want to try with dummy data, you can use the "--use_dummy_data" option.

## Run on local after pip install
```
python run_electricity.py --cluster_mode local
```

## Run on yarn cluster for yarn-client mode after pip install
```
python run_electricity.py --cluster_mode yarn
```

## Options
* `--cluster_mode` The mode for the Spark cluster. local or yarn. Default to be `local`. You can refer to OrcaContext documents [here](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) for details.
* `--num_workers` The number of workers to be used in the cluster. You can change it depending on your own cluster setting. Default to be 2.
* `--cores` "The number of cpu cores you want to use on each node. Default to be 4.
* `--memory` The memory you want to use on each node. Default to be 10g
* `--data_dir` The directory of electricity data file.
* `--use_dummy_data` Whether to use dummy data. Default to be False.
* `--smoke` Whether to run smoke test. Smoke test run 1 iteration for each stage and run 2 iterations alternative training. Default to be False.
* `--predict_local` You should enable predict_local if want to run distributed training on yarn and run distributed inference on local."
* `--num_predict_cores` The number of cores you want to use for prediction on local. You should only parse this arg if you have set predict_local to true.
* `--num_predict_workers` The number of workers you want to use for prediction on local. You should only parse this arg if you have set predict_local to true.