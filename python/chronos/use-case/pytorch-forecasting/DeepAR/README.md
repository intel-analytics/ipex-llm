# Use Nano to help pytorch-forecasting improve the training speed of DeepAR model
Nano can help a 3rd party time series lib to improve the performance (both training and inferencing) and accuracy. This use-case shows nano can easily help pytorch-forecasting speed up the training of DeepAR model.

## Prepare the environment
We recommend you to use conda to prepare the environment, especially if you want to run on a yarn cluster:
```bash
conda create -n my_env python=3.7 # "my_env" is conda environment name, you can use any name you like.
conda activate my_env
pip install --pre --upgrade bigdl-chronos[all]
pip install pytorch_forecasting  # note: you need to install a version >= 0.10.0
```
Please refer to [Chronos Install Guide](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html#install)

## Prepare data
We use ``pytorch_forecasting.data.example.generate_ar_data`` to generate our data automatically.

## Run the example
```bash
bigdl-nano-init python deepar.py
```

## Changes to use BigDL Nano
- Change `from pytorch_lightning import Trainer` to `from bigdl.chronos.pytorch import TSTrainer`
- Set `gpus=0` in TSTrainer
- Set `num_processes=8` in TSTrainer and set `batch_size = 64 // num_processes`

## Results
In an experimental platform, the training speed of DeepAR model using nano Trainer is 6 times the speed of the training without nano Trainer. We can see that the training speed is significantly improved.
