# Running Friesian listwise example

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
pip install --pre --upgrade bigdl-friesian[train]
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
(Worker pid=1921) Epoch 3/3
 3/10 [========>.....................] - ETA: 0s - root_mean_squared_error: 1.1129 - loss: 1.2385 - regularization_loss: 0.0000e+00 - total_loss: 1.2385
 7/10 [====================>.........] - ETA: 0s - root_mean_squared_error: 1.1080 - loss: 1.2277 - regularization_loss: 0.0000e+00 - total_loss: 1.2277
10/10 [==============================] - 0s 34ms/step - root_mean_squared_error: 1.1051 - loss: 1.2188 - regularization_loss: 0.0000e+00 - total_loss: 1.2188
1/5 [=====>........................] - ETA: 1s - root_mean_squared_error: 1.1029 - loss: 1.2163 - regularization_loss: 0.0000e+00 - total_loss: 1.2163
3/5 [=================>............] - ETA: 0s - root_mean_squared_error: 1.1021 - loss: 1.2145 - regularization_loss: 0.0000e+00 - total_loss: 1.2145
4/5 [=======================>......] - ETA: 0s - root_mean_squared_error: 1.1001 - loss: 1.2103 - regularization_loss: 0.0000e+00 - total_loss: 1.2103
Stopping orca context
```

#### YARN
```bash
python basic_ranking.py --data_dir hdfs://path/to/ml-1m --cluster_mode yarn --executor_cores 8 --executor_memory 50g --num_executor 2
```
