# Train an NCF model on MovieLens 
This example demonstrates how to use BigDL Friesian to train a [NCF](https://dl.acm.org/doi/10.1145/3038912.3052569) (Neural Collaborative Filtering) model using [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset.

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster.
```
conda create -n bigdl python=3.7  # "bigdl" is the conda environment name, you can use any name you like.
conda activate bigdl
pip install tensorflow==2.6.0
pip install pandas
pip install --pre --upgrade bigdl-friesian[train]
```

## Train NCF model
* Spark local, example command:
```bash
python ncf_train.py \
    --executor_cores 4 \
    --executor_memory 4g
```

* Spark standalone, example command:
```bash
python ncf_train.py \
    --cluster_mode standalone \
    --master spark://master-url:port \
    --executor_cores 4 \
    --executor_memory 4g \
    --num_executors 2
```

* Spark yarn client mode, example command:
```bash
python ncf_train.py \
    --cluster_mode yarn \
    --executor_cores 4 \
    --executor_memory 4g \
    --num_executors 2 \
    --model_dir /hdfs/path/to/save/ncf/model
```

__Options:__
* `cluster_mode`: The cluster mode to run the training, one of local, yarn, standalone or spark-submit. Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `executor_cores`: The number of cores to use on each executor. Default to be 8.
* `executor_memory`: The amount of memory to allocate on each executor. Default to be 4g.
* `num_executors`: The number of executors to use in the cluster. Default to be 2.
* `driver_cores`: The number of cores to use for the driver. Default to be 4.
* `driver_memory`: The amount of memory to allocate for the driver. Default to be 4g.
* `backend`: The backend of TF2 Estimator, either ray or spark. Default to be ray.
* `model_dir`: The directory to save the trained model. Default to be the current working directory. For yarn cluster, you need to provide an HDFS path.
* `data_dir`: The directory for the movielens data. Default to be `./movielens`. If the data is not detected, it will be automatically downloaded.
* `lr`: The learning rate to train the model. Default to be 0.001.
* `epochs`: The number of epochs to train the model. Default to be 5.
* `batch_size`: The batch size to train the model. Default to be 8000.