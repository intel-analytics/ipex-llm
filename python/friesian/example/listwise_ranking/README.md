# Running Friesian listwise example


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

### Train

#### Local
```bash
python listwise_ranking.py --data_dir /path/to/ml-1m
```
Result:
```bash
34/37 [==========================>...] - ETA: 0s - ndcg_metric: 0.9192 - root_mean_squared_error: 3.2804 - loss: 4.4367 - regularization_loss: 0.0000e+00 - total_loss: 4.4367
35/37 [===========================>..] - ETA: 0s - ndcg_metric: 0.9192 - root_mean_squared_error: 3.2845 - loss: 4.4367 - regularization_loss: 0.0000e+00 - total_loss: 4.4367
36/37 [============================>.] - ETA: 0s - ndcg_metric: 0.9192 - root_mean_squared_error: 3.2821 - loss: 4.4366 - regularization_loss: 0.0000e+00 - total_loss: 4.4366
37/37 [==============================] - ETA: 0s - ndcg_metric: 0.9192 - root_mean_squared_error: 3.2831 - loss: 4.4363 - regularization_loss: 0.0000e+00 - total_loss: 4.4363
37/37 [==============================] - 5s 148ms/step - ndcg_metric: 0.9192 - root_mean_squared_error: 3.2831 - loss: 4.4361 - regularization_loss: 0.0000e+00 - total_loss: 4.4361 - val_ndcg_metric: 0.9156 - val_root_mean_squared_error: 3.1994 - val_loss: 4.5184 - val_regularization_loss: 0.0000e+00 - val_total_loss: 4.5184
Stopping orca context
```

#### YARN
```bash
python listwise_ranking.py --data_dir hdfs://path/to/ml-1m --cluster_mode yarn --executor_cores 8 --executor_memory 50g --num_executor 2
```

Result:
```bash
(Worker pid=93823, ip=172.16.0.113) 2022-05-31 10:49:09.719951: W tensorflow/core/framework/dataset.cc:768] Inp37/37 [==============================] - 4s 113ms/step - ndcg_metric: 0.9186 - root_mean_squared_error: 3.2923 - loss: 2.2265 - regularization_loss: 0.0000e+00 - total_loss: 2.2265 - val_ndcg_metric: 0.9161 - val_root_mean_squared_error: 3.3426 - val_loss: 4.5400 - val_regularization_loss: 0.0000e+00 - val_total_loss: 4.5400
(Worker pid=93823, ip=172.16.0.113) Epoch 29/30
37/37 [==============================] - 4s 114ms/step - ndcg_metric: 0.9185 - root_mean_squared_error: 3.3016 - loss: 2.2236 - regularization_loss: 0.0000e+00 - total_loss: 2.2236 - val_ndcg_metric: 0.9159 - val_root_mean_squared_error: 3.4069 - val_loss: 4.5383 - val_regularization_loss: 0.0000e+00 - val_total_loss: 4.5383
(Worker pid=93823, ip=172.16.0.113) Epoch 30/30
37/37 [==============================] - ETA: 0s - ndcg_metric: 0.9188 - root_mean_squared_error: 3.3209 - loss: 2.2214 - regularization_loss: 0.0000e+00 - total_loss: 2.2214
(Worker pid=93824, ip=172.16.0.113) 2022-05-31 10:49:18.175228: W tensorflow/core/framework/dataset.cc:768] Input of GeneratorDatasetOp::Dataset will not be optimized because the dataset does not implement the AsGraphDefInternal() method needed to apply optimizations.
Stopping orca context
```
