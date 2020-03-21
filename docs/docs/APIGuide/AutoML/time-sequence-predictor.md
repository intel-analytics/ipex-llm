# TimeSequencePredictor

`TimeSequencePredictor` can be used to train a pipeline (including feature engineering and model) for 
automated time series forecasting in a distributed way. AutoML is applied for searching the best 
set of features as well as model hyper-parameters.

## Methods

### \_\_init\_\_

```python
tsp = TimeSequencePredictor(name="automl",
                            logs_dir="~/zoo_automl_logs",
                            future_seq_len=1,
                            dt_col="datetime",
                            target_col="value",
                            extra_features_col=None,
                            drop_missing=True,)
```

#### Arguments

* **name**: Name of the experiment.

* **logs_dir**: Where the automl tune logs file located.

* **future_seq_len**: Integer. The future sequence length to be predicted. The default value is 1.

* **dt_col**: The name of datetime column of the input data frame.

* **target_col**: The name of target column to be predicted of the input data frame.

* **extra_features_col**: The name of extra features column that needs for prediction of the input data frame.

* **drop_missing**: Boolean. Whether to drop missing values of the input data frame.


### fit

Train a pipeline for time series forecasting. It will return a `TimeSequencePipeline` object.

```python
tsp.fit(self,
        input_df,
        validation_df=None,
        metric="mse",
        recipe=SmokeRecipe(),
        mc=False,
        resources_per_trial={"cpu": 2},
        distributed=False
        )
```

#### Arguments

* **input_df**: Input time series data frame. It could look like:
          
    |datetime|value|...|
    | --------|----- | ---|
    |2019-06-06|1.2|...|
    |2019-06-07|2.3|...|

* **validation_df**: validation data frame. It should have the same columns with `input_df`.

* **metric**: String. Metric used for train and validation. Available values are "mean_squared_error" or "mean_absolute_error".

* **recipe**: A Recipe object. Various recipes covers different search space and stopping criteria. Default is `SmokeRecipe`. 
              Available recipes are `SmokeRecipe`, `RandomRecipe`, `GridRandomRecipe` and `BayesRecipe`.
              
* **resources_per_trial**: Machine resources to allocate per trial, e.g. `{"cpu": 64, "gpu": 8}`.

* **distributed**: Boolean. Whether to run in distributed mode. Default is False. 


