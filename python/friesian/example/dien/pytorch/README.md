# Train DIEN using the Amazon book review dataset
This folder showcases how to use BigDL Friesian to preprocess and train a [DIEN](https://arxiv.org/pdf/1809.03672.pdf) model. 
Model definition is based on [here](https://github.com/mouna99/dien) and
[Amazon Book Reviews](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz) dataset is used in this example.

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster (yarn-client mode only).
```
conda create -n bigdl python=3.7  # "bigdl" is the conda environment name, you can use any name you like.
conda activate bigdl
pip install pytorch=1.11.0
pip install --pre --upgrade bigdl-friesian
```

## Prepare the data
Data is preprocessed with ``../dien_preprocessing.py`` on yarn cluster.


## Train DIEN
* Spark local:
```bash
python dien_train.py \
    --executor_cores 8 \
    --executor_memory 50g \
    --batch_size 128 \
    --data_dir /path/to/the/folder/to/save/preprocessed/parquet/files \
```

* Spark standalone, example command:
```bash
python dien_train.py \
    --cluster_mode standalone \
    --master spark://master-url:port \
    --executor_cores 8 \
    --executor_memory 50g \
    --num_executors 2 \
    --batch_size 128 \
    --data_dir /path/to/the/folder/to/save/preprocessed/parquet/files
```

* Spark yarn client mode, example command:
```bash
python dien_train.py \
    --cluster_mode yarn \
    --executor_cores 8 \
    --executor_memory 50g \
    --num_executors 2 \
    --batch_size 512 \
    --data_dir /path/to/the/folder/to/save/preprocessed/parquet/files
```

__Options:__
* `data_dir`: __Required.__ The path to preprocessed data, either a local path or an HDFS path.
* `cluster_mode`: The cluster mode to run the data preprocessing, one of local, yarn, standalone or spark-submit. Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `executor_cores`: The number of cores to use on each node. Default to be 48.
* `executor_memory`: The amount of memory to allocate on each node. Default to be 160g.
* `num_executors`: The number of nodes to use in the cluster. Default to be 8.
* `driver_cores`: The number of cores to use for the driver. Default to be 4.
* `driver_memory`: The amount of memory to allocate for the driver. Default to be 36g.

## Performance
After 100 epochs' training, training accuracy is 0.73931, training AUC is 0.71903; test accurac is 0.68421, and test AUC is 0.65303. The training is conducted on yarn cluster, costing 1 hour and 31 minutes.
