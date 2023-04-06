# Use Chronos Forecasters on Electricity dataset

In this use-case, we will use Chronos Forecasters(TCN) of tensorflow backend on electricity dataset. Electricity dataset is a widely used public dataset for Time series forecasting in engineering validation and researching.

For API docs of TCNForecaster, please refer to
[TCNForecaster API Document](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html#tcnforecaster)

## Prepare the environment

We recommend you to use conda to prepare the environment.

```bash
conda create -n my_env python=3.7 setuptools=58.0.4 # "my_env" is conda environment name, you can use any name you like.
conda activate my_env
pip install --pre --upgrade bigdl-chronos[tensorflow]
source bigdl-nano-init # accelerate the environment
```

For more detailed information, please refer to [Chronos Install Guide](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/install.html)

## Prepare data

We are using the **Electricity** data with the preprocessing aligned with [this paper](https://arxiv.org/abs/2106.13008).

Download link: [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/) or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing).

You only need to download `electricity.csv`.

## Run the example

```bash
export OMP_NUM_THREADS=1
taskset -c 0 python tcn-tf.py # for tcn forecaster
```

## Output

### TCNForecaster
After you run the code, the training process is shown like this:
```bash
Epoch 2/30
550/550 [==============================] - 96s 174ms/step - loss: 0.3519 - mse: 0.3519
```

After training 30 epochs, MSE is shown like this,
```bash
MSE is: 0.34341344
```

and the inference latency is shown like this:
```bash
Inference latency(s) is: 0.033265095392684164
```
