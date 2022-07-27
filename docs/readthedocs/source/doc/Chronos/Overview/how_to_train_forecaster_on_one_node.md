# How-to Train Forecaster on One Node

1. Before you train the forecaster, the forecaster must be created through `Forecaster.from_tsdataset` or directly.

​       Please refer to [How to create a Forecaster]()

2. Train forecaster by calling `fit`.

   ##### (1) Call fit while having no validation data.

   Input the following 3 parameters: `data`, `epochs`, and `batch_size`.

   ```python
       forecaster = TCNForecaster(past_seq_len=24,                                                                      future_seq_len=5,
                                  input_feature_num=1,
                                  output_feature_num=1,
                                  …)
       forecaster.fit(train_data, epochs=2)
   ```

   The validation_step in the training loop will be skipped.

   `fit` has no return value in this way.

   Among them, the meaning and default values of parameters are as follows:

   ​    `data`: The `data` support following formats:

   ​        | 1. a numpy ndarray tuple (x, y):

   ​        | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim

   ​        | should be the same as `past_seq_len` and `input_feature_num`.

   ​        | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim

   ​        | should be the same as `future_seq_len` and `output_feature_num`.

   ​        | 2. a xshard item:

   ​        | each partition can be a dictionary of {'x': x, 'y': y}, where x and y's shape

   ​        | should follow the shape stated before.

   ​        | 3. pytorch dataloader:

   ​        | the dataloader should return x, y in each iteration with the shape as following:

   ​        | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim

   ​        | should be the same as `past_seq_len` and `input_feature_num`.

   ​        | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim

   ​        | should be the same as `future_seq_len` and `output_feature_num`.

   ​        | 4. A `bigdl.chronos.data.tsdataset.TSDataset` instance:

   ​        | Forecaster will automatically process the TSDataset.

   ​        | By default, TSDataset will be transformed to a pytorch dataloader,

   ​        | which is memory-friendly while a little bit slower.

   ​        | Users may call `roll` on the TSDataset before calling `fit`

   ​        | Then the training speed will be faster but will consume more memory.

   ​    `epochs`: Number of epochs you want to train. The value defaults to 1.

   ​    `batch_size`: Number of batch size you want to train. The value defaults to 32.

   ​        If you input a pytorch dataloader for `data`, the `batch_size` will follow the

   ​        `batch_size` setted in `data`.if the forecaster is distributed, the `batch_size` will be

   ​        evenly distributed to all workers.

   ##### (2) Call fit while having validation data.

   Input the following 6 parameters: `data`, `validation_data`, `epochs`, `batch_size`, `validation_mode`, and `earlystop_patience`.

   ```python
       forecaster = TCNForecaster(past_seq_len=24,                                                                      future_seq_len=5,
                                  input_feature_num=1,
                                  output_feature_num=1,
                                  …)
       train_loss = forecaster.fit(train_data, val_data,                                                                 validation_mode='earlystop',
                                   earlystop_patience=6, epochs=50)
   ```

   The validation_step in the training loop will be executed. And in this way, you need to input `validation_mode` to select the operation you want.

   `fit` will return a dict recording the average validation loss of each epoch in this way.

   Among them, the meaning and default values of the rest parameters are as follows:

   ​    `validation_data`: Validation sample for validation loop. Defaults to 'None'.

   ​        The `validation_data` support following formats:

   ​        | 1. a numpy ndarray tuple (x, y):

   ​        | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim

   ​        | should be the same as `past_seq_len` and `input_feature_num`.

   ​        | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim

   ​        | should be the same as `future_seq_len` and `output_feature_num`.

   ​        | 2. pytorch dataloader:

   ​        | the dataloader should return x, y in each iteration with the shape as following:

   ​        | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim

   ​        | should be the same as `past_seq_len` and `input_feature_num`.

   ​        | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim

   ​        | should be the same as `future_seq_len` and `output_feature_num`.

   ​    `validation_mode`:  A str represent the operation mode while having '`validation_data`'.

   ​        Defaults to 'output'. The validation_mode includes the following types:

   ​        | 1. output:

   ​        | If you choose 'output' for validation_mode, it will return a dict that records the

   ​        | average validation loss of each epoch.

   ​        | 2. earlystop:

   ​        | Monitor the val_loss and stop training when it stops improving.

   ​    `earlystop_patience`: Number of checks with no improvement after which training will

   ​        be stopped. It takes effect when '`validation_mode`' is 'earlystop'. Under the default

   ​        configuration, one check happens after every training epoch.

3. Further reading

   distributed: When your data volume is large and difficult to complete on a single machine, we also provide distributed training, please refer to [train model distributed on a cluster]()

​       forecaster.load:
