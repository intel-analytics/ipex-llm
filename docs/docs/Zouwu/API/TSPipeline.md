#TSPipeline

A pipeline for time series forecasting. 
```python
from zoo.zouwu.autots.forecast import TSPipeline
```
__Note:__

- TSPipeline can be obtained from AutoTSTrainer or loaded from saved file.
- train_df and validation_df in fit/evalute/predict are data frames. An exmaple data frame looks like below.

  |datetime|value|extra_feature_1|extra_feature_2|
  | --------|----- |---| ---|
  |2019-06-06|1.2|1|2|
  |2019-06-07|2.3|0|2|

## Methods

### fit
This is usually for incremental fitting, and doesn't involve AutoML.

```python
fit(input_df,validation_df=None,uncertainty: bool = False,epochs=1,**user_config)
```
#### Arguments
* **input_df**: the input dataframe
* **validation_df**: the validation dataframe
* **uncertainty**: whether to calculate uncertainty
* **epochs**: number of epochs to train
* **user_config**: user configurations

### predict
```python
predict(input_df) 
```
#### Arguments
* **input_df**: the input dataframe
* **return**: the forecast results

### evaluate
```python
evaluate(input_df,metrics=["mse"],multioutput='raw_values')
```
#### Arguments
* **input_df**: the input dataframe
* **metrics**: the evaluation metrics
* **multioutput**: output mode of multiple output, whether to aggregate
* **return**: the evaluation results
 

---
Load and Save a TSPipeline can be used in below way.

```python
 from zoo.zouwu.autots.forecast import TSPipeline
 loaded_ppl = TSPipeline.load(file)
 # ... do sth. e.g. incremental fitting
 loaded_ppl.save(another_file)
```

### load 

load is a static method. 

```python
load(pipeline_file)
```
#### Arguments
* **pipeline_file**: the pipeline file

### save

```python
save(pipeline_file)
```
#### Arguments
* **pipeline_file**: the pipeline file

 

