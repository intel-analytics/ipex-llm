## Install

Zouw needs below requirements to run. 

* Python 3.6 or 3.7
* PySpark 2.4.3
* Ray 0.8.4
* Tensorflow 1.15.0
* aiohttp
* setproctitle
* scikit-learn >=0.20.0
* psutil
* requests
* featuretools
* pandas
* Note that Keras is not needed to use Zouwu. But if you have Keras installed, make sure it is Keras 1.2.2. Other verisons might cause unexpected problems.

You can install above python dependencies manually, or install using below command. 
```python
pip install analytics-zoo[automl]
```

## Using built-in forecast models

The built-in forecast models are all derived from [tfpark.KerasModels](https://analytics-zoo.github.io/master/#APIGuide/TFPark/model/). 

 1.To start, you need to create a forecast model first. Specify **target_dim** and **feature_dim** in constructor. 

*  ```target_dim```: dimension of target output
*  ```feature_dim```: dimension of input feature

Refer to API doc for detailed explaination of all arguments for each forecast model.

Below are some example code to create forecast models.

```python
#import forecast models
from zoo.zouwu.model.forecast import LSTMForecaster
from zoo.zouwu.model.forecast import MTNetForecaster

#build a lstm forecast model
lstm_forecaster = LSTMForecaster(target_dim=1, 
                      feature_dim=4)
                      
#build a mtnet forecast model
mtnet_forecaster = MTNetForecaster(target_dim=1,
                        feature_dim=4,
                        lb_long_steps=1,
                        lb_long_stepsize=3,
                        ar_window_size=2,
                        cnn_kernel_size=2)
```
 
2.Use ```forecaster.fit/evalute/predict``` in the same way as [tfpark.KerasModel](https://analytics-zoo.github.io/master/#APIGuide/TFPark/model/)



3.For univariant forecasting (i.e. to predict one series at a time), you can use either **LSTMForecaster** or **MTNetForecaster**. The input data shape for `fit/evaluation/predict` should match the arguments you used to create the forecaster. Specifically:

* **X** shape should be ```(num of samples, lookback, feature_dim)```
* **Y** shape should be ```(num of samples, target_dim)```
* Where, ```feature_dim``` is the number of features as specified in Forecaster constructors. ```lookback``` is the number of time steps you want to look back in history. ```target_dim``` is the number of series to forecast at the same time as specified in Forecaster constructors and should be 1 here. If you want to do multi-step forecasting and use the second dimension as no. of steps to look forward, you won't get error but the performance may be uncertain and we don't recommend using that way.


 4.For multivariant forecasting (i.e. to predict several series at the same time), you have to use **MTNetForecaster**. The input data shape should meet below criteria.  

* **X** shape should be ```(num of samples, lookback, feature_dim)```
* **Y** shape should be ```(num of samples, target_dim)``` 
* Where ```lookback``` should equal ```(lb_long_steps+1) * lb_long_stepsize```, where ```lb_long_steps``` and ```lb_long_stepsize``` are as specified in ```MTNetForecaster``` constructor. ```target_dim``` should equal number of series in input.

---

## Using AutoTS

The automated training in zouwu is built upon [Analytics Zoo AutoML module](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/automl) (refer to [AutoML ProgrammingGuide](https://analytics-zoo.github.io/master/#ProgrammingGuide/AutoML/overview/) and [AutoML APIGuide](https://analytics-zoo.github.io/master/#APIGuide/AutoML/time-sequence-predictor/) for details), which uses [Ray Tune](https://github.com/ray-project/ray/tree/master/python/ray/tune) for hyper parameter tuning and runs on [Analytics Zoo RayOnSpark](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/).  

The general workflow using automated training contains below two steps. 

1. create a `AutoTSTrainer` to train a `TSPipeline`, save it to file to use later or elsewhere if you wish.
2. use `TSPipeline` to do prediction, evaluation, and incremental fitting as well. 

---

You'll need ```RayOnSpark``` for training with ```AutoTSTrainer```, so you have to init it before auto training, and stop it after training is completed. Note RayOnSpark is not needed if you just use TSPipeline for inference, evaluation or incremental training. 

* init RayOnSpark in local mode

```python
from zoo import init_spark_on_local
from zoo.ray import RayContext
sc = init_spark_on_local(cores=4)
ray_ctx = RayContext(sc=sc)
ray_ctx.init()
```
* init RayOnSpark on yarn

```python
from zoo import init_spark_on_yarn
from zoo.ray import RayContext
slave_num = 2
sc = init_spark_on_yarn(
        hadoop_conf=args.hadoop_conf,
        conda_name="ray36",
        num_executor=slave_num,
        executor_cores=4,
        executor_memory="8g ",
        driver_memory="2g",
        driver_cores=4,
        extra_executor_memory_for_ray="10g")
ray_ctx = RayContext(sc=sc, object_store_memory="5g")
ray_ctx.init()
```

* After training, stop RayOnSpark. 

```python
   ray_ctx.stop()
```

---

Both AutoTSTrainer and TSPipeline accepts data frames as input. An exmaple data frame looks like below.

  |datetime|value|extra_feature_1|extra_feature_2|
  | --------|----- |---| ---|
  |2019-06-06|1.2|1|2|
  |2019-06-07|2.3|0|2|

---

1.To create an AutoTSTrainer. Specify below arguments in constructor. See below example.

* ```dt_col```: the column specifying datetime 
* ```target_col```: target column to predict
* ```horizon``` : num of steps to look forward 
* ```extra_feature_col```: a list of columns which are also included in input as features except target column

```python
 from zoo.zouwu.autots.forecast import AutoTSTrainer

 trainer = AutoTSTrainer(dt_col="datetime",
                         target_col="value",
                         horizon=1,
                         extra_features_col=None)

```
 
2.Use ```AutoTSTrainer.fit``` on train data and validation data. A TSPipeline will be returned. 
```python
 ts_pipeline = trainer.fit(train_df, validation_df)
```


3.Use ```TSPipeline.fit/evaluate/predict``` to train pipeline (incremental fitting), evaluate or predict. 
```python
 #incremental fitting
 ts_pipeline.fit(new_train_df, new_val_df, epochs=10)
 #evaluate
 ts_pipeline.evalute(val_df)
 ts_pipeline.predict(test_df) 

```

4.Use ```TSPipeline.save/load``` to load from file or save to file. 

```python
 from zoo.zouwu.autots.forecast import TSPipeline
 loaded_ppl = TSPipeline.load(file)
 # ... do sth. e.g. incremental fitting
 loaded_ppl.save(another_file)
```
 

