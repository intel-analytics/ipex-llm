# Multi-task Recommendation with BigDL
In addition to providing a personalized recommendation, recommendation systems need to output diverse 
predictions to meet the needs of real-world applications, such as user click-through rates and browsing (or watching) time predictions for products.
This example demonstrates how to use the [MMoE](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) or [PLE](https://dl.acm.org/doi/pdf/10.1145/3383313.3412236?casa_token=8fchWD8CHc0AAAAA:2cyP8EwkhIUlSFPRpfCGHahTddki0OEjDxfbUFMkXY5fU0FNtkvRzmYloJtLowFmL1en88FRFY4Q) model to implement multi-task recommendations with large-scale data.

## Prepare environments
We highly recommend you use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment, especially if you want to run on a yarn cluster. 
```
conda create -n bigdl python=3.7 #bigdl is conda environment name, you can set another name you like.
conda activate bigdl
pip install bigdl-orca[ray]
pip install bigdl-friesian
pip install tensorflow==2.9.1
pip install deepctr[cpu]
```
Refer to [this document](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/python.html#install) for more installation guides.

## Data Preparation
In this example, a news dataset is used to demonstrate the training and testing process. 
Each row contains several feature values, timestamps and two labels. Using the timestamp to divide the training and testing sets. 
The click prediction (classification) and duration time prediction (regression) are two output targets. Original data examples are as follows:
```angular2html
+----------+----------+-------------------+----------+----------+-------------+-----+--------+------+-------+--------+--------+------+------+-------------------+-------+-------------+--------------------+
|   user_id|article_id|          expo_time|net_status|flush_nums|exop_position|click|duration|device|     os|province|    city|   age|gender|              ctime|img_num|        cat_1|               cat_2|
+----------+----------+-------------------+----------+----------+-------------+-----+--------+------+-------+--------+--------+------+------+-------------------+-------+-------------+--------------------+
|1000541010| 464467760|2021-06-30 09:57:14|         2|         0|           13|    1|      28|V2054A|Android|Shanghai|Shanghai|A_0_24|female|2021-06-29 14:46:43|      3|Entertainment| Entertainment/Stars|
|1000541010| 463850913|2021-06-30 09:57:14|         2|         0|           15|    0|       0|V2054A|Android|Shanghai|Shanghai|A_0_24|female|2021-06-27 22:29:13|     11|     Fashions|Fashions/Female F...|
|1000541010| 464022440|2021-06-30 09:57:14|         2|         0|           17|    0|       0|V2054A|Android|Shanghai|Shanghai|A_0_24|female|2021-06-28 12:22:54|      7|        Rural|Rural/Agriculture...|
|1000541010| 464586545|2021-06-30 09:58:31|         2|         1|           20|    0|       0|V2054A|Android|Shanghai|Shanghai|A_0_24|female|2021-06-29 13:25:06|      5|Entertainment| Entertainment/Stars|
|1000541010| 465352885|2021-07-03 18:13:03|         5|         0|           18|    0|       0|V2054A|Android|Shanghai|Shanghai|A_0_24|female|2021-07-02 10:43:51|     18|Entertainment| Entertainment/Stars|
+----------+----------+-------------------+----------+----------+-------------+-----+--------+------+-------+--------+--------+------+------+-------------------+-------+-------------+--------------------+
```

With the built-in high-level preprocessing operations in FeatureTable, we can easily perform distributed pre-processing for large-scale data.
The details of pre-processing can be found [here](https://github.com/intel-analytics/BigDL/blob/main/apps/wide-deep-recommendation/feature_engineering.ipynb). Examples of processed data are as follows:

```angular2html
+-------------------+-----+--------+-------------------+-----------+-----+-------+----------+----------+----------+-------------+------+---+--------+----+---+------+-----+
|          expo_time|click|duration|              ctime|    img_num|cat_2|user_id|article_id|net_status|flush_nums|exop_position|device| os|province|city|age|gender|cat_1|
+-------------------+-----+--------+-------------------+-----------+-----+-------+----------+----------+----------+-------------+------+---+--------+----+---+------+-----+
|2021-06-30 09:57:14|    1|      28|2021-06-29 14:46:43|0.016574586|   60|  14089|     87717|         4|        73|         1003|    36|  2|      38| 308|  5|     1|    5|
|2021-06-30 09:57:14|    0|       0|2021-06-27 22:29:13| 0.06077348|   47|  14089|     35684|         4|        73|           43|    36|  2|      38| 308|  5|     1|   32|
|2021-06-30 09:57:14|    0|       0|2021-06-28 12:22:54|0.038674034|  157|  14089|     20413|         4|        73|          363|    36|  2|      38| 308|  5|     1|   20|
|2021-06-30 09:58:31|    0|       0|2021-06-29 13:25:06|0.027624309|   60|  14089|     15410|         4|       312|          848|    36|  2|      38| 308|  5|     1|    5|
|2021-07-03 18:13:03|    0|       0|2021-07-02 10:43:51| 0.09944751|   60|  14089|     81707|         2|        73|          313|    36|  2|      38| 308|  5|     1|    5|
+-------------------+-----+--------+-------------------+-----------+-----+-------+----------+----------+----------+-------------+------+---+--------+----+---+------+-----+
```
Data pre-processing command:
```bash
python data_processing.py \
    --input_path  path/to/input/dataset \
    --output_path path/to/save/processed/dataset \
    --cluster_mode local \
    --executor_cores 8 \
    --executor_memory 24g \
    --num_executors 4 \
    --driver_cores 2 \
    --driver_memory 24g
```

__Options for data_processing:__
* `input_path`: The path to input dataset.
* `output_path`: The path to save processed dataset.
* `cluster_mode`: The cluster mode, such as local, yarn, standalone or spark-submit. Default to be local. 
* `master`: The master url, only used when cluster mode is standalone. Default to be None. 
* `executor_cores`: The executor core number. Default to be 8.
* `executor_memory`: The executor memory. Default to be 24g.
* `num_executors`: The number of executors. Default to be 4.
* `driver_cores`: The driver core number. Default to be 2. 
* `driver_memory`: The driver memory. Default to be 24g.

__NOTE:__ 
When the *cluster_mode* is yarn, *input_path* and *output_path* can be HDFS paths. 

## Train and test Multi-task models
After data preprocessing, training MMoE or PlE model as follows:
```bash
python run_multi_task.py \
    --do_train \
    --model_type mmoe\
    --train_data_path path/to/training/dataset \
    --test_data_path path/to/testing/dataset \
    --model_save_path path/to/save/the/trained/model \
    --cluster_mode local \
    --executor_cores 8 \
    --executor_memory 24g \
    --num_executors 4 \
    --driver_cores 2 \
    --driver_memory 24g
```

Evaluate Results as follows:
```bash
python run_multi_task.py \
    --do_test \
    --model_type mmoe\
    --test_data_path path/to/testing/dataset \
    --model_save_path path/to/save/the/trained/model \
    --cluster_mode local \
    --executor_cores 8 \
    --executor_memory 24g \
    --num_executors 4 \
    --driver_cores 2 \
    --driver_memory 24g
```

__Options for data_processing:__
* `do_train`: To start training model.
* `do_test`: To start test model.
* `model_type`: The multi task model, mmoe or ple. Default to be mmoe.
* `train_data_path`: The path to training dataset.
* `test_data_path`: The path to testing dataset.
* `model_save_path`: The path to save model.
* `cluster_mode`: The cluster mode, such as local, yarn, standalone or spark-submit. Default to be local. 
* `master`: The master url, only used when cluster mode is standalone. Default to be None. 
* `executor_cores`: The executor core number. Default to be 8.
* `executor_memory`: The executor memory. Default to be 24g.
* `num_executors`: The number of executors. Default to be 4.
* `driver_cores`: The driver core number. Default to be 2. 
* `driver_memory`: The driver memory. Default to be 24g.

__NOTE:__ 
When the *cluster_mode* is yarn, *train_data_path*, *test_data_path* ans *model_save_path* can be HDFS paths. 
