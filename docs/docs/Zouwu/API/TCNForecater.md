# TCNForecaster

## Introduction

Temporal Convolutional Networks (TCN) is a neural network that use convolutional architecture rather than recurrent networks. It supports multi-step and multi-variant cases. Causal Convolutions enables large scale parallel computing which makes TCN has less inference time than RNN based model such as LSTM.

## Method

### Arguments

- **`num_channels`**: Specify the convolutional layer filter number in TCN's encoder.
- **`kernel_size`**: Specify convolutional layer filter height in TCN's encoder.
- **`dropout`**: Specify the dropout close possibility (i.e. the close possibility to a neuron).
- **`optimizer`**: Specify the optimizer used for training. This value defaults to Adam.
- **`lr`**: Specify the learning rate. This value defaults to 0.001.

### \_\_init\_\_

```python
TCNForecaster(num_channels=[30]*8,
              kernel_size=7,
              dropout=0.2,
              optimizer="Adam",
              lr=0.001,
              )
```

### fit

```python
fit(x, y, epochs=1, metric="mse", batch_size=32)
```

- **`x`**: A numpy array with size (num_samples, input_time_steps, input_feature_dim).
- **`y`**: A numpy array with size (num_samples, output_time_steps, output_feature_dim).
- **`epochs`**: Number of epochs you want to train.
- **`batch_size`**: Number of batch size you want to train.
- **`metric`**: The metric for training data.

### evaluate

```python
evaluate(x, y, metric=['mse'])
```

- **`x`**: A numpy array with size (num_samples, input_time_steps, input_feature_dim).
- **`y`**: A numpy array with size (num_samples, output_time_steps, output_feature_dim).
- **`metric`**: The metric for test/valid data.

### predict

```python
predict(x)
```

- **`x`**: A numpy array with size (num_samples, input_time_steps, input_feature_dim).

### save

```python
save(checkpoint_file)
```

- **`checkpoint_file`**: The location you want to save the forecaster

### restore

```python
restore(checkpoint_file)
```

- **`checkpoint_file`**: The location you want to save the forecaster