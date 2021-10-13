# Train DIEN using the Amazon book review dataset
This folder showcases how to preprocess and train a [DIEN](https://arxiv.org/pdf/1809.03672.pdf) model on Analytics Zoo. 
Model definition is based on [ai-matrix](https://github.com/alibaba/ai-matrix/tree/master/macro_benchmark/DIEN)
[Amazon book review](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz) and [meta_books](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz) dataset to be used in this example.

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

## Preprocess data  
* Spark local, example command:
```bash
python dien_preprocessing.py \
    --executor_cores 8 \
    --executor_memory 50g \
    --input_meta /path/to/the/folder/of/meta_books \
    --input_transaction /path/to/the/folder/of/review_data\
    --output /path/to/the/folder/to/save/preprocessed/parquet_files 
```

* Spark standalone, example command:
```bash
python dien_preprocessing.py \
    --cluster_mode standalone \
    --master spark://master-url:port \
    --executor_cores 40 \
    --executor_memory 240g \
    --num_executor 8 \
    --input_meta /path/to/the/folder/of/meta_books \
    --input_transaction /path/to/the/folder/of/review_data\
    --output /path/to/the/folder/to/save/preprocessed/parquet_files 
```

* Spark yarn client mode, example command:
```bash
python dien_preprocessing.py \
    --cluster_mode yarn \
    --executor_cores 40 \
    --executor_memory 240g \
    --input_meta /path/to/the/folder/of/meta_books \
    --input_transaction /path/to/the/folder/of/review_data\
    --output /path/to/the/folder/to/save/preprocessed/parquet_files 
```

__Options:__
* `cluster_mode`: The cluster mode to run the data preprocessing, one of local, yarn or standalone. Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `executor_cores`: The number of cores to use on each node. 
* `executor_memory`: The amount of memory to allocate on each node. 
* `num_nodes`: The number of nodes to use in the cluster. 
* `driver_cores`: The number of cores to use for the driver. 
* `driver_memory`: The amount of memory to allocate for the driver.
* `input_meta`: The path to the folder of meta_books jason files, either a local path or an HDFS path.
* `input_transaction`: The path to the folder of review_data jason files, either a local path or an HDFS path.
* `output`: The path to save the preprocessed data to parquet files. HDFS path is recommended.

## Train DIEN
* Spark local:
```bash
python dien_train.py \
    --executor_cores 8 \
    --executor_memory 50g \
    --batch_size 128 \
    --data_dir /path/to/the/folder/to/save/preprocessed/parquet_files \
    --model_dir /path/to/the/folder/to/save/trained/model 
```

* Spark standalone, example command:
```bash
python dien_train.py \
    --cluster_mode standalone \
    --master spark://master-url:port \
    --executor_cores 8 \
    --executor_memory 240g \
    --num_executor 8 \
    --batch_size 128 \
    --data_dir /path/to/the/folder/to/save/preprocessed/parquet_files \
    --model_dir /path/to/the/folder/to/save/trained/model 
```

* Spark yarn client mode, example command:
```bash
python dien_train.py \
    --cluster_mode yarn \
    --executor_cores 8 \
    --executor_memory 240g \
    --batch_size 128 \
    --data_dir /path/to/the/folder/to/save/preprocessed/parquet_files \
    --model_dir /path/to/the/folder/to/save/trained/model 
```

__Options:__
* `cluster_mode`: The cluster mode to run the data preprocessing, one of local, yarn or standalone. Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `executor_cores`: The number of cores to use on each node. Default to be 48.
* `executor_memory`: The amount of memory to allocate on each node. Default to be 240g.
* `num_nodes`: The number of nodes to use in the cluster. Default to be 40.
* `driver_cores`: The number of cores to use for the driver. Default to be 4.
* `driver_memory`: The amount of memory to allocate for the driver. Default to be 36g.
* `batch_size`: Batch size.
* `data_dir`: The path of preprocessed data.
* `data_dir`: The path to save trained model.
