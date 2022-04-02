# Use Nano to help pytorch-forecasting improve the training speed of DeepAR model
Nano can help a 3rd party time series lib to improve the performance (both training and inferencing) and accuracy. This use-case shows nano can easily help pytorch-forecasting speed up the training of DeepAR model.

## Prepare the environment
We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:
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

## Results
The training of DeepAR model using nano Trainer can reach an average speed of 1.9 iter/s, while the speed of training without nano Trainer is around 1.4 iter/s in average. We can see that the training speed is significantly improved.

* CPU info
    * model name: Intel(R) Xeon(R) Gold 6252 CPU @ 2.10GHz
    * cpu cores: 48