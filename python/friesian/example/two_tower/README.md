# Train a two tower model using recsys data
This example demonstrates how to use our BigDL Friesian to train a two tower model using [Twitter Recsys Challenge 2021 data](https://recsys-twitter.com/data/show-downloads#).

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster (yarn-client mode only).
```
conda create -n bigdl python=3.7  # "bigdl" is the conda environment name, you can use any name you like.
conda activate bigdl
pip install --pre --upgrade bigdl-friesian
```
## Preprocess data
You can download the full Twitter dataset from [here](https://recsys-twitter.com/data/show-downloads#) and then follow the [WideAndDeep Preprocessing] to preprocess the orginal data.

## Training 2 tower model
* Spark local, we can use some sample data to have a trial, example command:
```bash
python train_2tower.py \
    --executor_cores 8 \
    --executor_memory 50g \
    --data_dir /path/to/the/folder/of/sample_data \
    --model_dir /path/to/the/folder/to/save/trained_model 
```

* Spark standalone, example command:
```bash
python train_2tower.py \
    --cluster_mode standalone \
    --master spark://master-url:port \
    --executor_cores 56 \
    --executor_memory 240g \
    --num_executor 8 \
    --data_dir /path/to/the/folder/of/sample_data \
    --model_dir /path/to/the/folder/to/save/trained_model 
```

* Spark yarn client mode, example command:
```bash
python train_2tower.py \
    --cluster_mode yarn \
    --executor_cores 56 \
    --executor_memory 240g \
    --data_dir /path/to/the/folder/of/sample_data \
    --model_dir /path/to/the/folder/to/save/trained_model 
```

__Options:__
* `cluster_mode`: The cluster mode to run the data preprocessing, one of local, yarn or standalone. Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `executor_cores`: The number of cores to use on each node. Default to be 48.
* `executor_memory`: The amount of memory to allocate on each node. Default to be 240g.
* `num_nodes`: The number of nodes to use in the cluster. Default to be 8.
* `driver_cores`: The number of cores to use for the driver. Default to be 4.
* `driver_memory`: The amount of memory to allocate for the driver. Default to be 36g.
* `data_dir`: The input data directory as well as output of embedding reindex tables.
* `model_dir`: The output, including model for trained models and stats to stroage reindex dicts and min_max.pkl
