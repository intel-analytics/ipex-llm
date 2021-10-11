#AutoTSTrainer

Chronos AutoTSTrainer is used to train a TSPipeline for forecasting using AutoML.

It is built upon [Analytics Zoo AutoML module](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/automl) (refer to [AutoML ProgrammingGuide](https://analytics-zoo.github.io/master/#ProgrammingGuide/AutoML/overview/) and [AutoML APIGuide](https://analytics-zoo.github.io/master/#APIGuide/AutoML/time-sequence-predictor/) for details), which uses [Ray Tune](https://github.com/ray-project/ray/tree/master/python/ray/tune) for hyper parameter tuning and runs on [Analytics Zoo RayOnSpark](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/).

## Methods

### \_\_init\_\_
```python
from zoo.chronos.autots.deprecated.forecast import AutoTSTrainer

trainer = AutoTSTrainer(dt_col="datetime",
                         target_col="value",
                         horizon=1,
                         extra_features_col=None,
                         search_alg=None,
                         search_alg_params=None,
                         scheduler=None,
                         scheduler_params=None,)

```

* **dt_col**: the column specifying datetime
* **target_col**: target column to predict
* **horizon** : num of steps to look forward
* **extra_feature_col**: a list of columns which are also included in input as features except target column
* **search_alg**: Optional(str). The search algorithm to use. We only support "bayesopt" and "skopt" for now.
                The default search_alg is None and variants will be generated according to the search method in search space.
* **search_alg_params**: Optional(Dict). params of search_alg.
* **scheduler**: Optional(str). Scheduler name. Allowed scheduler names are "fifo", "async_hyperband",
    "asynchyperband", "median_stopping_rule", "medianstopping", "hyperband", "hb_bohb", "pbt". The default scheduler is "fifo".
* **scheduler_params**: Optional(Dict). Necessary params of scheduler.

### fit

```python 
fit(train_df,
    validation_df=None,
    metric="mse",
    recipe: Recipe = SmokeRecipe(),
    uncertainty: bool = False)
 ```

* **train_df**: the input dataframe (as pandas.dataframe)
* **validation_df**: the validation dataframe (as pandas.dataframe)
* **recipe**: the configuration of searching, refer to definition in [automl.config.recipe](../../APIGuide/AutoML/recipe.md)
* **metric**: the evaluation metric to optimize
* **uncertainty**: whether to enable uncertainty calculation (will output an uncertainty sigma)
* **return**: a TSPipeline
 
__Note:__

train_df and validation_df are data frames. An exmaple data frame looks like below.

  |datetime|value|extra_feature_1|extra_feature_2|
  | --------|----- |---| ---|
  |2019-06-06|1.2|1|2|
  |2019-06-07|2.3|0|2|


 

