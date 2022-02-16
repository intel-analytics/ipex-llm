## Chronos AutoTS Quickstart

In this guide, we will show you how to use AutoTS for automated time series forecasting.

The general workflow using AutoTS contains below two steps. 

1. create a `AutoTSTrainer` to train a `TSPipeline`, save it to file to use later or elsewhere if you wish.
2. use `TSPipeline` to do prediction, evaluation, and incremental fitting as well. 

Refer to [AutoTS notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_autots_forecasting.ipynb) for demonstration how to use AutoTS to build a time series forcasting pipeline. 

Refer to [AutoTS API Guide](https://analytics-zoo.github.io/master/#Chronos/API/AutoTSTrainer/) for more details of AutoTS APIs.

---
### **Step 0: Prepare environment**
Chronos AutoTS needs below requirements to run.

* python 3.6 or 3.7
* pySpark
* analytics-zoo
* tensorflow>=1.15.0,<2.0.0
* h5py==2.10.0
* ray[tune]==1.9.2
* psutil
* aiohttp
* setproctitle
* pandas
* scikit-learn>=0.20.0,<=0.22.0
* requests

You can install above python dependencies manually. But we strongly recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run automated training on a yarn cluster (yarn-client mode only).
```bash
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo[automl]==0.9.0.dev0 # or above
```

### **Step 1: Init Orca Context**
You'll need ```RayOnSpark``` for training with ```AutoTSTrainer```, so you should init an orca context with `init_ray_on_spark=True` before auto training, and stop it after training is completed. Note orca context is not needed if you just use TSPipeline for inference, evaluation or incremental training. 
```python
from zoo.orca import init_orca_context, stop_orca_context

# run in local mode
init_orca_context(cluster_mode="local", cores=4, memory='2g', num_nodes=1, init_ray_on_spark=True)

# run in yarn client mode
init_orca_context(cluster_mode="yarn-client", 
                  num_nodes=2, cores=2, 
                  driver_memory="6g", driver_cores=4, 
                  conda_name='zoo', 
                  extra_memory_for_ray="10g", 
                  object_store_memory='5g')
```
* Reference: [Orca Context](https://analytics-zoo.github.io/master/#Orca/context/)

### **Step 2: Create an AutoTSTrainer**
To create an AutoTSTrainer. Specify below arguments in constructor. See below example.

* ```dt_col```: the column specifying datetime 
* ```target_col```: target column to predict
* ```horizon``` : num of steps to look forward 
* ```extra_feature_col```: a list of columns which are also included in input as features except target column
* ```search_alg```: Optional(str). The search algorithm to use. We only support "bayesopt" and "skopt" for now.
                The default search_alg is None and variants will be generated according to the search method in search space.
* ```search_alg_params```: Optional(Dict). params of search_alg.
* ```scheduler```: Optional(str). Scheduler name. Allowed scheduler names are "fifo", "async_hyperband",
    "asynchyperband", "median_stopping_rule", "medianstopping", "hyperband", "hb_bohb", "pbt". The default scheduler is "fifo".
* ```scheduler_params```: Optional(Dict). Necessary params of scheduler.

```python
from zoo.chronos.autots.deprecated.forecast import AutoTSTrainer

trainer = AutoTSTrainer(dt_col="datetime",
                        target_col="value",
                        horizon=1,
                        extra_features_col=None)

```
### **Step 3: Fit with AutoTSTrainer**

Use ```AutoTSTrainer.fit``` on train data and validation data. A TSPipeline will be returned. 

```python
ts_pipeline = trainer.fit(train_df, validation_df)
```

Both AutoTSTrainer and TSPipeline accepts data frames as input. An exmaple data frame looks like below.

|datetime|value|extra_feature_1|extra_feature_2|
| --------|----- |---| ---|
|2019-06-06|1.2|1|2|
|2019-06-07|2.3|0|2|

**Note:** you should call `stop_orca_context()` when your distributed automated training finishes.

For visualization, please refer to [here](../../ProgrammingGuide/AutoML/visualization.md).

### **Step 4: Further deployment with TSPipeline**
Use ```TSPipeline.fit/evaluate/predict``` to train pipeline (incremental fitting), evaluate or predict. 
```python
#incremental fitting
ts_pipeline.fit(new_train_df, new_val_df, epochs=10)
#evaluate
ts_pipeline.evalute(val_df)
ts_pipeline.predict(test_df) 

```

Use ```TSPipeline.save/load``` to load from file or save to file. 

```python
from zoo.chronos.autots.deprecated.forecast import TSPipeline
loaded_ppl = TSPipeline.load(file)
# ... do sth. e.g. incremental fitting
loaded_ppl.save(another_file)
```


