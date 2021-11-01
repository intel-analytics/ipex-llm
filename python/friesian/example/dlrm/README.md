# Preprocess the Criteo dataset for DLRM Model
This example demonstrates how to use BigDL Friesian to preprocess the 
[Criteo](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) dataset to be used for [DLRM](https://arxiv.org/abs/1906.00091) model training.

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster (yarn-client mode only).
```
conda create -n bigdl python=3.7  # "bigdl" is the conda environment name, you can use any name you like.
conda activate bigdl
pip install --pre --upgrade bigdl-friesian
```

__Note:__ As we test, Spark 3 will have performance benefit over the default Spark 2.4.6 and you can download the BigDL package built with Spark 3.0.0 from [here](https://sourceforge.net/projects/analytics-zoo/files/zoo-py/).

## Prepare the data
You can download the full __1TB__ Click Logs dataset from [here](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/), which includes data of 24 days (day_0 to day_23) with 4,373,472,329 records in total.

After you download the files, convert them to parquet files with the name `day_x.parquet` (x=0-23), and put all parquet files in one folder.
- The first 23 days (day_0 to day_22) are used for DLRM training with 4,195,197,692 records in total.
- The first half (89,137,319 records in total) of the last day (day_23) is used for test. To prepare the test dataset, you need to split the first half of day_23 into a new file (e.g. using command `head -n 89137319 day_23 > day_23_test`) and finally convert to parquet files with the name `day_23_test.parquet` under the same folder with the train parquet files.

If you want to use some sample data for test, you can download `dac_sample` from [here](https://labs.criteo.com/2014/02/download-dataset/), unzip and rename it to day_0 and convert to `day_0.parquet`.

## Running command
* Spark local, we can use the first (several) day(s) or the sample data to have a trial, example command:
```bash
python dlrm_preprocessing.py \
    --cores 36 \
    --memory 50g \
    --days 0-0 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --output_folder /path/to/the/folder/to/save/preprocessed/parquet_files/and/string_index \
    --frequency_limit 15
```

* Spark standalone, example command to run on the full Criteo dataset:
```bash
python dlrm_preprocessing.py \
    --cluster_mode standalone \
    --master spark://master-url:port \
    --cores 56 \
    --memory 240g \
    --num_nodes 8 \
    --days 0-23 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --output_folder /path/to/the/folder/to/save/preprocessed/parquet_files/and/string_index \
    --frequency_limit 15
```

* Spark yarn client mode, example command to run on the full Criteo dataset:
```bash
python dlrm_preprocessing.py \
    --cluster_mode yarn \
    --cores 56 \
    --memory 240g \
    --num_nodes 8 \
    --days 0-23 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --output_folder /path/to/the/folder/to/save/preprocessed/parquet_files/and/string_index \
    --frequency_limit 15
```

__Options:__
* `cluster_mode`: The cluster mode to run the data preprocessing, one of local, yarn or standalone. Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `cores`: The number of cores to use on each node. Default to be 48.
* `memory`: The amount of memory to allocate on each node. Default to be 240g.
* `num_nodes`: The number of nodes to use in the cluster. Default to be 8.
* `driver_cores`: The number of cores to use for the driver. Default to be 4.
* `driver_memory`: The amount of memory to allocate for the driver. Default to be 36g.
* `days`: The day range for data preprocessing, such as 0-23 for the full Criteo dataset, 0-0 for the first day, 0-1 for the first two days, etc. Default to be 0-23.
* `frequency_limit`: Categories with frequency below this value will be omitted from encoding. We recommend using 15 when you preprocess the full 1TB dataset. Default to be 15.
* `input_folder`: This option is __required__. The path to the folder of parquet files, either a local path or an HDFS path.
* `output_folder`: The path to save the preprocessed data and the generated string indices to parquet files. HDFS path is recommended. 
The generated string indices will be saved to `output_folder/[colname].parquet` respectively. For the full 1TB dataset, preprocessed train data 
will be saved to `output_folder/saved_data` and test data will be saved to `output_folder/saved_data_test`. Default to be None. If not specified, the program will only do the computation but not save the preprocessed data and the generated string indices.
