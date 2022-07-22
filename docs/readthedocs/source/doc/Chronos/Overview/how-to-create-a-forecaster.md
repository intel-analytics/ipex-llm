## How to create a Forecaster
### 1. Forecaster
Before creating a Forecaster,you need to understand the four parameters `past_seq_len`, `future_seq_len`, `input_feature_num`, `output_feature_num`, which represent the time step and feature column, As shown below.
![](../Image/forecast-RR.png "time series")
 * **past_seq_len**: Sampled input length, represents the history time step. (i.e. lookback)
 * **future_seq_len**: Sampled output length, represents the output time step.(i.e. horizon)
 * **input_feature_num**: All feature column(s), including extra feature column(s) and target column(s).
 * **output_feature_num**: Only target column(s).
After determining your above parameters, you can create a forecaster next.
```note
Some forecaster models may only require some parameters, such as
LSTMforecaster without specifying future_seq_len, because lstm is a single-step model.
NBeatsForecaster only needs to specify past_seq_len and future_seq_len.
```

### 2. create a forecaster.
 We provide two methods for creating Forecaster.
 #### 1. Create a Forecaster using Forecaster.from_tsdataset(Highly recommended)
 > :note: `from_tsdataset` requires input of a TSDataset instance. TSDataset is a built-in time series preprocessing class, More info, Please refer to [how to use TSDataset and get_public_dataset](xxxx).
 
 If your tsdataset has used the `roll` or `to_torch_data_loader` methods,
 you do not need to specify `past_seq_len` and `future_seq_len` for from_tsdataset, otherwise you must specify both.
 The simplest usage of "from_tsdataset" is given below. More info, please refer to [from_tsdataset](xxxx).
 ```python
 from bigdl.chronos.forecaster import TCNForecaster
 from bigdl.chronos.data import TSDataset
 tsdataset = TSDataset.from_pandas(df, ...)  # df is a pandas.DataFrame
 tcn = TCNForecaster.from_tsdataset(tsdataset,
                                    past_seq_len=past_seq_len,
                                    future_seq_len=future_seq_len)
 tcn.fit(tsdataset)
 ```

 #### 2. Create a forecaster directly
 You can also create TCNForecaster directly and you can choose to use TSDataset or other custom data preprocessing methods.
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
 training: After you have created the forecaster, you can train your model, Please refer to [how to train]()
 distributed: When your data volume is large and it is difficult to complete on a single machine, we also provide distributed training, please refer to [train model distributed on a cluster]()
 forecaster.load: 