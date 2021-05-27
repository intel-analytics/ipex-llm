# TCNForecaster

## Introduction

Temporal Convolutional Networks (TCN) is a neural network that use convolutional architecture rather than recurrent networks. It supports multi-step and multi-variant cases. Causal Convolutions enables large scale parallel computing which makes TCN has less inference time than RNN based model such as LSTM.

## Method

### Arguments

- **`past_seq_len`**: Specify the history time steps (i.e. lookback).
- **`future_seq_len`**: Specify the output time steps (i.e. horizon).
- **`input_feature_num`**: Specify the feature dimension.
- **`output_feature_num`**: Specify the output dimension.
- **`num_channels`**: Specify the convolutional layer filter number in TCN's encoder. This value defaults to \[30\]*8.
- **`kernel_size`**: Specify convolutional layer filter height in TCN's encoder. This value defaults to 7.
- **`dropout`**: Specify the dropout close possibility (i.e. the close possibility to a neuron). This value defaults to 0.2.
- **`optimizer`**: Specify the optimizer used for training. This value defaults to "Adam".
- **`loss`**: Specify the loss function used for training. This value defaults to "mse". You can choose from "mse", "mae" and "huber_loss".
- **`lr`**: Specify the learning rate. This value defaults to 0.001.

### \_\_init\_\_

```python
TCNForecaster(past_seq_len,
              future_seq_len,
              input_feature_num,
              output_feature_num,
              num_channels=[30]*8,
              kernel_size=7,
              dropout=0.2,
              optimizer="Adam",
              loss="mse",
              lr=0.001)
```

### fit

```python
fit(x, y, validation_data=None, epochs=1, metric="mse", batch_size=32)
```

- **`x`**: A numpy array with shape (num_samples, lookback, feature_dim). lookback and feature_dim should be the same as `past_seq_len` and `input_feature_num`.
- **`y`**: A numpy array with shape (num_samples, horizon, target_dim). horizon and target_dim should be the same as `future_seq_len` and `output_feature_num`.
- **`validation_data`**: A tuple (x_valid, y_valid) as validation data. Default to None.
- **`epochs`**: Number of epochs you want to train.
- **`batch_size`**: Number of batch size you want to train.
- **`metric`**: The metric for training data.

### evaluate

```python
evaluate(x, y, metrics=['mse'], multioutput="raw_values")
```

- **`x`**: A numpy array with shape (num_samples, lookback, feature_dim).
- **`y`**: A numpy array with shape (num_samples, horizon, target_dim).
- **`metrics`**: A list contains metrics for test/valid data.
- **`multioutput`**: Defines aggregating of multiple output values. String in ['raw_values', 'uniform_average']. The value defaults to 'raw_values'.
### evaluate_with_onnx

```python
evaluate_with_onnx(x, y, metrics=['mse'], dirname=None, multioutput="raw_values")
```

- **`x`**: A numpy array with shape (num_samples, lookback, feature_dim).
- **`y`**: A numpy array with shape (num_samples, horizon, target_dim).
- **`metrics`**: A list contains metrics for test/valid data.
- **`dirname`**: The directory to save onnx model file. This value defaults to None for no saving file.
- **`multioutput`**: Defines aggregating of multiple output values. String in ['raw_values', 'uniform_average']. The value defaults to 'raw_values'.

### predict

```python
predict(x)
```

- **`x`**: A numpy array with shape (num_samples, lookback, feature_dim).

### predict_with_onnx

```python
predict_with_onnx(x, dirname=None)
```

- **`x`**: A numpy array with shape (num_samples, lookback, feature_dim).
- **`dirname`**: The directory to save onnx model file. This value defaults to None for no saving file.

### save

```python
save(checkpoint_file)
```

- **`checkpoint_file`**: The location you want to save the forecaster.

### restore

```python
restore(checkpoint_file)
```

- **`checkpoint_file`**: The checkpoint file location you want to load the forecaster.