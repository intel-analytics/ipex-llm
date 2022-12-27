# Train a WideAndDeep Model on the Criteo dataset
This example demonstrates how to use BigDL Friesian to preprocess the 
[Criteo](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) dataset and train the [WideAndDeep](https://arxiv.org/abs/1606.07792) model in a distributed fashion.

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster.
```
conda create -n bigdl python=3.7  # "bigdl" is the conda environment name, you can use any name you like.
conda activate bigdl
pip install tensorflow==2.9.0
pip install --pre --upgrade bigdl-friesian[train]
```

## Prepare the data
You can download the full __1TB__ Click Logs dataset from [here](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/), which includes data of 24 days (day_0 to day_23) with 4,373,472,329 records in total.

After you download the files, convert them to parquet files with the name `day_x.parquet` (x=0-23), and put all parquet files in one folder. You may use the script `csv_to_parquet.py` provided in this directory to convert the data of each day to parquet.
- The first 23 days (day_0 to day_22) are used for WND training with 4,195,197,692 records in total.
- The first half (89,137,319 records in total) of the last day (day_23) is used for test. To prepare the test dataset, you need to split the first half of day_23 into a new file (e.g. using command `head -n 89137319 day_23 > day_23_test`) and finally convert to parquet files with the name `day_23_test.parquet` under the same folder with the train parquet files.

If you want to use some sample data for test, you can download `dac_sample` from [here](https://labs.criteo.com/2014/02/download-dataset/), unzip and convert `dac_sample.txt` to parquet with name `day_0.parquet`.

## Data Preprocessing
* Spark local, we can use the first (several) day(s) or the sample data to have a trial, example command:
```bash
python wnd_preprocessing.py \
    --executor_cores 36 \
    --executor_memory 50g \
    --days 0-0 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --output_folder /path/to/the/folder/to/save/preprocessed/parquet_files \
    --frequency_limit 15 \
    --cross_sizes 10000,10000
```

* Spark standalone, example command to run on the full Criteo dataset:
```bash
python wnd_preprocessing.py \
    --cluster_mode standalone \
    --master spark://master-url:port \
    --executor_cores 56 \
    --executor_memory 240g \
    --num_executors 8 \
    --days 0-23 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --output_folder /path/to/the/folder/to/save/preprocessed/parquet_files \
    --frequency_limit 15 \
    --cross_sizes 10000,10000
```

* Spark yarn client mode, example command to run on the full Criteo dataset:
```bash
python wnd_preprocessing.py \
    --cluster_mode yarn \
    --executor_cores 56 \
    --executor_memory 240g \
    --num_executors 8 \
    --days 0-23 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --output_folder /path/to/the/folder/to/save/preprocessed/parquet_files \
    --frequency_limit 15 \
    --cross_sizes 10000,10000
```

__Options:__
* `input_folder`: The path to the folder of parquet files, either a local path or an HDFS path.
* `output_folder`: The path to save the preprocessed data to parquet files and meta data. HDFS path is recommended for yarn cluster_mode.
* `cluster_mode`: The cluster mode to run the data preprocessing, one of local, yarn, standalone or spark-submit. Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `executor_cores`: The number of cores to use on each executor. Default to be 48.
* `executor_memory`: The amount of memory to allocate on each executor. Default to be 160g.
* `num_executors`: The number of executors to use in the cluster. Default to be 8.
* `driver_cores`: The number of cores to use for the driver. Default to be 4.
* `driver_memory`: The amount of memory to allocate for the driver. Default to be 36g.
* `days`: The day range for data preprocessing, such as 0-23 for the full Criteo dataset, 0-0 for the first day, 0-1 for the first two days, etc. Default to be 0-23.
* `frequency_limit`: Categories with frequency below this value will be omitted from encoding. We recommend using 15 when you preprocess the full 1TB dataset. Default to be 15.
* `cross_sizes`: The bucket sizes for cross columns (`c14-c15` and `c16-c17`) separated by comma. Default to be 10000,10000. Please pay attention that there must NOT be a blank space between the two numbers.

## Model training
* Spark local, example command:
```bash
python wnd_train.py \
    --executor_cores 36 \
    --executor_memory 50g \
    --data_dir /path/to/the/folder/of/preprocessed/parquet_files \
    --model_dir ./wnd_model
```

* Spark standalone, example command:
```bash
python wnd_train.py \
    --cluster_mode standalone \
    --master spark://master-url:port \
    --executor_cores 56 \
    --executor_memory 240g \
    --num_executors 8 \
    --data_dir /path/to/the/folder/of/preprocessed/parquet_files \
    --model /path/to/save/the/trained/model
```

* Spark yarn client mode, example command:
```bash
python wnd_train.py \
    --cluster_mode yarn \
    --executor_cores 56 \
    --executor_memory 240g \
    --num_executors 8 \
    --data_dir /path/to/the/folder/of/preprocessed/parquet_files \
    --model /path/to/save/the/trained/model
```

__Options:__
* `data_dir`: The path to the folder of preprocessed parquet files and meta data, either a local path or an HDFS path.
* `cluster_mode`: The cluster mode to run the data preprocessing, one of local, yarn, standalone or spark-submit. Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `executor_cores`: The number of cores to use on each executor. Default to be 48.
* `executor_memory`: The amount of memory to allocate on each executor. Default to be 30g.
* `num_executors`: The number of executors to use in the cluster. Default to be 8.
* `driver_cores`: The number of cores to use for the driver. Default to be 4.
* `driver_memory`: The amount of memory to allocate for the driver. Default to be 36g.
* `model_dir`: The path to saved the trained model, either a local path or an HDFS path. Default to be "./wnd_model".
* `batch_size`: The batch size to train the model. Default to be 1024.
* `epoch`: The number of epochs to train the model. Default to be 2.
* `learning_rate`: The learning rate to train the model. Default to be 0.0001.
