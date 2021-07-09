# Preprocess the Amazon book review dataset for DIEN Model
This example demonstrates how to use Analytics Zoo Friesian to preprocess the 
[Amazon book review](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz) and [meta_books](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz) dataset to be used for [DIEN](https://arxiv.org/pdf/1809.03672.pdf) model training.

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster (yarn-client mode only).
```
conda create -n zoo python=3.7  # "zoo" is the conda environment name, you can use any name you like.
conda activate zoo
pip install --pre --upgrade analytics-zoo
```

## Prepare the data
   1. Down load meta_books data from [here](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz). 
   2. Down load full book_review data from [here](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz) which contains 22,507,155 records, or you can start from the [small dataset](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz) which contains 8,898,041 records.

## Running command
* Spark local, we can use the first several days or some sample data to have a trial, example command:
```bash
python dien_preprocessing.py \
    --executor_cores 36 \
    --executor_memory 50g \
    --input_meta /path/to/the/folder/of/meta_books \
    --input_transaction /path/to/the/folder/of/review_data\
    --output /path/to/the/folder/to/save/preprocessed/parquet_files 
```

* Spark standalone, example command to run on the full Criteo dataset:
```bash
python dien_preprocessing.py \
    --cluster_mode standalone \
    --master spark://master-url:port \
    --executor_cores 56 \
    --executor_memory 240g \
    --num_executor 8 \
    --input_meta /path/to/the/folder/of/meta_books \
    --input_transaction /path/to/the/folder/of/review_data\
    --output /path/to/the/folder/to/save/preprocessed/parquet_files 
```

* Spark yarn client mode, example command to run on the full Criteo dataset:
```bash
python dien_preprocessing.py \
    --cluster_mode yarn \
    --executor_cores 56 \
    --executor_memory 240g \
    --input_meta /path/to/the/folder/of/meta_books \
    --input_transaction /path/to/the/folder/of/review_data\
    --output /path/to/the/folder/to/save/preprocessed/parquet_files 
```

__Options:__
* `cluster_mode`: The cluster mode to run the data preprocessing, one of local, yarn or standalone. Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `executor_cores`: The number of cores to use on each node. Default to be 48.
* `executor_memory`: The amount of memory to allocate on each node. Default to be 240g.
* `num_nodes`: The number of nodes to use in the cluster. Default to be 8.
* `driver_cores`: The number of cores to use for the driver. Default to be 4.
* `driver_memory`: The amount of memory to allocate for the driver. Default to be 36g.
* `input_meta`: The path to the folder of meta_books jason files, either a local path or an HDFS path.
* `input_transaction`: The path to the folder of review_data jason files, either a local path or an HDFS path.
* `output`: The path to save the preprocessed data to parquet files. HDFS path is recommended.
