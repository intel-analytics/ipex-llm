# Use Nano to help pytorch-forecasting improve the training speed of TFT model
Nano can help a 3rd party time series lib to improve the performance (both training and inferencing) and accuracy. This use-case shows nano can easily help pytorch-forecasting speed up the training of TFT (Temporal Fusion Transformers) model.

## Prepare the environment
We recommend you to use conda to prepare the environment, especially if you want to run on a yarn cluster:
```bash
conda create -n my_env python=3.7 # "my_env" is conda environment name, you can use any name you like.
conda activate my_env
pip install --pre --upgrade bigdl-chronos[all]
pip install pytorch_forecasting
pip install torch==1.11.0  # if your pytorch_forecasting version is 0.10.0 or above, you need to reinstall torch, otherwise you don't need this.
```
Please refer to [Chronos Install Guide](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html#install)

## Prepare data
We are using the [Stallion dataset from Kaggle](https://www.kaggle.com/datasets/utathya/future-volume-prediction). The data will be loaded as pandas dataframe automatically.

## Run the example
```bash
bigdl-nano-init python tft.py
```

## Changes to use BigDL Nano
- Change `from pytorch_lightning import Trainer` to `from bigdl.chronos.pytorch import TSTrainer`
- Set `gpus=0` in TSTrainer
- Set `num_processes=8` in TSTrainer and set `batch_size = 128 // num_processes`

## Results
In an experimental platform, the training speed of TFT model using nano Trainer is 3.5 times the speed of the training without nano Trainer. We can see that the training speed is significantly improved.
