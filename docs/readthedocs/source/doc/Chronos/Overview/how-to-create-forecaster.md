## How to create a Forecaster
### 1. Forecaster
Before creating a Forecaster,you need to understand the four parameters `past_seq_len`, `future_seq_len`, `input_feature_num`, `output_feature_num`, which represent the time step and feature column, As shown below.
![](../Image/forecast-RR.png "time series")

 * **past_seq_len**: Sampled input length, represents the history time step. (i.e. lookback)
 * **future_seq_len**: Sampled output length, represents the output time step.(i.e. horizon)
 * **input_feature_num**: All feature column(s), including extra feature column(s) and target column(s).
 * **output_feature_num**: Only target column(s).

After understanding these 4 parameters, the next step is to create a Forecaster.
```note
Some forecaster models may only require some parameters, such as
LSTMforecaster without specifying future_seq_len, because lstm is a single-step model.
```

### 2. create a forecaster.
We provide two ways to create a Forecaster.

#### Create a Forecaster using Forecaster.from_tsdataset(Highly recommended)
`from_tsdataset` is a classmethod, so you can call `Forecsater.from_tsdataset`, then input a `TSDataset` instance.
`TSDataset` is a built-in time series preprocessing class.
If your tsdataset has used the `roll` or `to_torch_data_loader` methods, you do not need to specify `past_seq_len` and `future_seq_len` for from_tsdataset, otherwise you must specify both.
You can also specify the hyperparameters of the model, such as lr, dropout etc. The simplest usage of "from_tsdataset" is given below.

```python
from bigdl.chronos.forecaster import TCNForecaster
from bigdl.chronos.data import TSDataset
tsdataset = TSDataset.from_pandas(df, ...)  # df is a pandas.DataFrame
# No call roll and to_torch_data_loader
tcn = TCNForecaster.from_tsdataset(tsdataset,
                                   past_seq_len=48,
                                   future_seq_len=5)
tcn.fit(tsdataset)

# Call roll or to_torch_dataloader, do not specify past_seq_len and future_seq_len
loader = tsdataset.to_torch_data_loader(...)
tcn = TCNForecaster.from_tsdataset(tsdataset)
tcn.fit(loader)
```

#### Create a forecaster directly
You can also create TCNForecaster directly, the parameters mentioned above still need to be specified.

```python
from bigdl.chronos.forecaster import TCNForecaster
# prepare dataset
timeseries = ...

tcn = TCNForecaster(past_seq_len=48,
                    future_seq_len=5,
                    input_feature_num=2,
                    output_feature_num=2)

tcn.fit(timeseries)
```

### 3. further reading
training: After you have created the forecaster, you can train your model.
distributed: When your data volume is large and it is difficult to complete on a single machine, we also provide distributed training.
load: When you have trained and saved the model locally, you can load a model.
