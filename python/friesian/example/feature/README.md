# Preprocess the Criteo dataset for DLRM Model
This example demonstrates how to use Analytics-zoo Friesian to preprocess the 
[Criteo](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) dataset for DLRM model.

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster(yarn-client mode only).
```
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install --pre --upgrade analytics-zoo
```

## Prepare the data
You can download Criteo data from <https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/>.
 
After you download the files, you should convert them to parquet files with the name day_x.parquet(x
=0-23), and put all parquet files in one folder.


## Preprocess
* Spark local, we can use the first several days to have a trial, example command
```bash
python dlrm_preprocessing.py \
    --executor_cores 36 \
    --executor_memory 50g \
    --days 0-1 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --frequency_limit 15
```

* Spark standalone, example command
```bash
python dlrm_preprocessing.py \
    --cluster_mode standalone \
    --master spark://master/url \
    --driver_cores 4 \
    --driver_memory 36g \
    --executor_cores 56 \
    --executor_memory 160g \
    --num_executor 8 \
    --days 0-23 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --frequency_limit 15
```

* Spark yarn client mode, example command
```bash
python dlrm_preprocessing.py \
    --cluster_mode yarn \
    --driver_cores 4 \
    --driver_memory 36g \
    --executor_cores 56 \
    --executor_memory 160g \
    --num_executor 8 \
    --days 0-23 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --frequency_limit 15
```

In the above commands
* cluster_mode: The cluster mode, such as local, yarn, or standalone. Default: local.
* master: The master URL, only used when cluster mode is standalone.
* executor_cores: The executor core number. Default: 48.
* executor_memory: The executor memory. Default: 160g.
* num_executor: The number of executors. Default: 8.
* driver_cores: The driver core number. Default: 4.
* driver_memory: The driver memory. Default: 36g.
* days: Day range for preprocessing, such as 0-23, 0-1.
* input_folder: Path to the folder of parquet files.
* frequency_limit: Categories with a count/frequency below frequency_limit will be omitted from
 the encoding. For instance, '15', '_c14:15,_c15:16', etc. We recommend using "15" when you
  preprocess the whole 1TB dataset. Default: 15.
