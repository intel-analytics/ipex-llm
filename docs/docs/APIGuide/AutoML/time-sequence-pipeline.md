# TimeSequencePipeline

`TimeSequencePipeline` integrates feature engineering and time sequence model into a data analysis pipeline.
You can get an `TimeSequencePipeline` object after calling `TimeSequencePredictor.fit()` or `load_ts_pipeline`.

## Methods

### load_ts_pipeline

```python
ts_pipeline = load_ts_pipeline(file)
```

#### Arguments

* **file**: saved ts pipeline file.


### describe

Since you may get your `ts_pipeline` from a saved file which is a result pipeline of `TimeSeqencePredictor`. You can use
`describe` method to get the initialization info for the `TimeSeqencePredictor`, including `future_seq_len`, `dt_col`, `target_col`, `extra_features_col`, `drop_missing`.

```python
ts_pipeline.describe()
```


### fit

Used for incremental fitting. Note that `fit` in `TimeSequencePipeline` doesn`t run in distributed mode.

```python
ts_pipeline.fit(input_df, validation_df=None, mc=False, epoch_num=20)
```

#### Arguments

* **input_df**: Input time series data frame. It should have the same datetime column, target column, 
                extra features columns (if there are) name with the names you got from `ts_pipeline.describe()`. 

* **validation_df**: The validation data frame. It should have the same columns with `input_df`.
         
* **mc**: Boolean. Whether to use Monte Carlo Dropout to predict with uncertainty. Default is False.
          You can refer to the paper [here](https://arxiv.org/abs/1709.01907) for more details about Monte Carlo Dropout. 

* **epoch_num**: Integer. Number of epochs to run incremental fitting.
                   
                   
### evaluate

```python
ts_pipeline.evaluate(input_df,
                     metrics=["mse"],
                     multioutput=`raw_values`)
```

#### Arguments

* **input_df**: Input data frame. It should have the same column names as the data frame you used to train your `TimeSequencePredictor`. 
                And you can get the information with `pipeline.describe()` 

* **metrics**: The metrics you want to use for evaluation. The available metrics are 
               `me`, `mae`, `mse`, `rmse`, `msle`, `r2`, `mpe`, `mape`, `mspe`, `smape`, `mdape` and `smdape`.
        
* **multioutput**: string in [`raw_values`, `uniform_average`]

                - `raw_values` : Returns a full set of errors.
                - `uniform_average` : Errors of all outputs are averaged with uniform weight.

### predict

```python
ts_pipeline.predict(input_df)
```

#### Arguments
* **input_df**: Input data frame to do prediction.It should have the same column names as the data frame you used to train your `TimeSequencePredictor`. 
                And you can get the information with `pipeline.describe()` 

            - a TFDataset object
            - A Numpy array (or array-like), or a list of arrays
               (in case the model has multiple inputs).
            - A dict mapping input names to the corresponding array/tensors,

### save

```python
ts_pipeline.save(ppl_file="my.ppl")
```

#### Arguments

* **ppl_file**: The file name you want to save your pipeline.


### config_save

```python
ts_pipeline.config_save(config_file=="my.json")
```

#### Arguments

* **config_file**: The config file name that you want to save all the pipeline configs in.

