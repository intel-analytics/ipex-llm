# Train a DeepFM model using recsys data
This example demonstrates how to train a xgb classification model in a distributed way using [Twitter Recsys Challenge 2021 data](https://recsys-twitter.com/data/show-downloads#).

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster (yarn-client mode only).
```
conda create -n bigdl python=3.7  # "bigdl" is the conda environment name, you can use any name you like.
conda activate bigdl
pip install --pre --upgrade bigdl-friesian
pip install xgboost
```

## Preprocess data
You can download the full Twitter dataset from [here](https://recsys-twitter.com/data/show-downloads#) and then follow the [WideAndDeep Preprocessing](https://github.com/intel-analytics/BigDL/tree/branch-2.0/python/friesian/example/wnd) to preprocess the orginal data.

## Training  tower model
* Spark local, we can use some sample data to have a trial, example command:
```bash
python xgb_train.py \
    --executor_cores 4 \
    --executor_memory 50g \
    --data_dir /path/to/the/folder/of/sample_data \
    --model_dir /path/to/the/folder/to/save/trained_model
```

* Spark yarn client mode, example command:
```bash
python xgb_train.py \
    --cluster_mode yarn \
    --num_executor 20 \
    --executor_cores 4 \
    --executor_memory 240g \
    --data_dir /path/to/the/folder/of/sample_data \
    --model_dir /path/to/the/folder/to/save/trained_model
```

### note for comparison:
best results from recsys data: Accuracy: 69.89; AUC: 76.77; params: {'tree_method': 'hist', 'eta': 0.3, 'gamma': 0.1, 'min_child_weight': 10, 'reg_lambda': 1, 'scale_pos_weight': 2, 'subsample': 1, 'objective': 'binary:logistic', 'max_depth': 12, 'num_round': 800}

__Options:__
* `cluster_mode`: The cluster mode to run the data preprocessing, one of local, spark submit or yarn Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `executor_cores`: The number of cores to use on each node. Default to be 48.
* `executor_memory`: The amount of memory to allocate on each node. Default to be 240g.
* `num_nodes`: The number of nodes to use in the cluster. Default to be 8.
* `driver_cores`: The number of cores to use for the driver. Default to be 4.
* `driver_memory`: The amount of memory to allocate for the driver. Default to be 36g.
* `data_dir`: The input data directory as well as output of embedding reindex tables.
* `model_dir`: The output, including model for trained models and stats to stroage reindex dicts and min_max.pkl


## references:
Tianqi Chen, Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. arXiv:1603.02754.