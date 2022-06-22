# Running Orca basic ranking example

This example is based on Tensorflow Recommenders example [basic ranking](https://www.tensorflow.org/recommenders/examples/basic_ranking).

## Environment

We recommend conda to set up your environment. You can install a conda distribution from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
if you haven't already.

```bash
conda create -n bigdl python==3.7
conda activate bigdl
pip install tensorflow pandas pyarrow pillow numpy
```

Then download and install latest nightly-build BigDL Friesian.
```bash
pip install --pre --upgrade bigdl-orca[ray]
```

## Training Data

Download MovieLens 1M Dataset [here](https://grouplens.org/datasets/movielens/1m/) and unzip the ml-1m.zip file.


## Running example

#### Local
```bash
python basic_ranking.py --data_dir /path/to/ml-1m
```
Result:
```bash
(Worker pid=23384) Epoch 3/3
 1/10 [==>...........................] - ETA: 0s - loss: 1.2286 - root_mean_squared_error: 1.1084
 5/10 [==============>...............] - ETA: 0s - loss: 1.2389 - root_mean_squared_error: 1.1131
 9/10 [==========================>...] - ETA: 0s - loss: 1.2308 - root_mean_squared_error: 1.1094
10/10 [==============================] - 0s 28ms/step - loss: 1.2316 - root_mean_squared_error: 1.1098
1/5 [=====>........................] - ETA: 5s - loss: 1.2971 - root_mean_squared_error: 1.1389
3/5 [=================>............] - ETA: 0s - loss: 1.2357 - root_mean_squared_error: 1.1116
4/5 [=======================>......] - ETA: 0s - loss: 1.2344 - root_mean_squared_error: 1.1110
Stopping orca context
```

#### YARN
```bash
python basic_ranking.py --data_dir hdfs://path/to/ml-1m --cluster_mode yarn --executor_cores 8 --executor_memory 50g --num_executor 2
```
