#AutoTSTrainer

Zouwu AutoTSTrainer is used to train a TSPipeline for forecasting using AutoML.

It is built upon [Analytics Zoo AutoML module](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/automl) (refer to [AutoML ProgrammingGuide](https://analytics-zoo.github.io/master/#ProgrammingGuide/AutoML/overview/) and [AutoML APIGuide](https://analytics-zoo.github.io/master/#APIGuide/AutoML/time-sequence-predictor/) for details), which uses [Ray Tune](https://github.com/ray-project/ray/tree/master/python/ray/tune) for hyper parameter tuning and runs on [Analytics Zoo RayOnSpark](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/).

## Methods

### \_\_init\_\_
```python
from zoo.zouwu.autots.forecast import AutoTSTrainer

trainer = AutoTSTrainer(dt_col="datetime",
                         target_col="value"
                         horizon=1,
                         extra_features_col=None)

```

* **dt_col**: the column specifying datetime
* **target_col**: target column to predict
* **horizon** : num of steps to look forward
* **extra_feature_col**: a list of columns which are also included in input as features except target column
 
### fit

```python 
fit(train_df,
    validation_df=None,
    metric="mse",
    recipe: Recipe = SmokeRecipe(),
    uncertainty: bool = False,
    distributed: bool = False,
    hdfs_url=None    validation_df)
 ```

* **train_df**: the input dataframe (as pandas.dataframe)
* **validation_df**: the validation dataframe (as pandas.dataframe)
* **recipe**: the configuration of searching, refer to definition in [automl.config.recipe](../../APIGuide/AutoML/recipe.md)
* **metric**: the evaluation metric to optimize
* **uncertainty**: whether to enable uncertainty calculation (will output an uncertainty sigma)
* **hdfs_url**: the hdfs_url to use for storing trail and intermediate results
* **distributed**: whether to enable distributed training
* **return**: a TSPipeline
 
__Note:__

train_df and validation_df are data frames. An exmaple data frame looks like below.

  |datetime|value|extra_feature_1|extra_feature_2|
  | --------|----- |---| ---|
  |2019-06-06|1.2|1|2|
  |2019-06-07|2.3|0|2|


 

