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
(Worker pid=6486, ip=172.16.0.135) Epoch 16/16
 1/23 [>.............................] - ETA: 15s - ndcg_metric: 0.9279 - root_mean_squared_error: 3.4357 - loss: 1751.7400 - regularization_loss: 0 
 2/23 [=>............................] - ETA: 14s - ndcg_metric: 0.9325 - root_mean_squared_error: 3.1181 - loss: 2724.6534 - regularization_loss: 0 
 3/23 [==>...........................] - ETA: 14s - ndcg_metric: 0.9337 - root_mean_squared_error: 3.2558 - loss: 2389.6213 - regularization_loss: 0 
 4/23 [====>.........................] - ETA: 13s - ndcg_metric: 0.9347 - root_mean_squared_error: 3.2409 - loss: 2274.2965 - regularization_loss: 0 
 5/23 [=====>........................] - ETA: 12s - ndcg_metric: 0.9342 - root_mean_squared_error: 3.1670 - loss: 2290.8057 - regularization_loss: 0 
 6/23 [======>.......................] - ETA: 12s - ndcg_metric: 0.9346 - root_mean_squared_error: 3.1463 - loss: 2235.8682 - regularization_loss: 0 
 7/23 [========>.....................] - ETA: 11s - ndcg_metric: 0.9339 - root_mean_squared_error: 3.2056 - loss: 2325.0197 - regularization_loss: 0 
 8/23 [=========>....................] - ETA: 11s - ndcg_metric: 0.9339 - root_mean_squared_error: 3.2423 - loss: 2285.4219 - regularization_loss: 0 
 9/23 [==========>...................] - ETA: 10s - ndcg_metric: 0.9343 - root_mean_squared_error: 3.2280 - loss: 2327.3718 - regularization_loss: 0
 10/23 [============>.................] - ETA: 9s - ndcg_metric: 0.9345 - root_mean_squared_error: 3.2278 - loss: 2338.7697 - regularization_loss: 0.
 11/23 [=============>................] - ETA: 9s - ndcg_metric: 0.9345 - root_mean_squared_error: 3.2058 - loss: 2315.0618 - regularization_loss: 0.
 12/23 [==============>...............] - ETA: 8s - ndcg_metric: 0.9344 - root_mean_squared_error: 3.1938 - loss: 2261.8331 - regularization_loss: 0.
 13/23 [===============>..............] - ETA: 7s - ndcg_metric: 0.9342 - root_mean_squared_error: 3.2703 - loss: 2245.5494 - regularization_loss: 0.
 14/23 [=================>............] - ETA: 6s - ndcg_metric: 0.9339 - root_mean_squared_error: 3.2671 - loss: 2280.3842 - regularization_loss: 0.
 15/23 [==================>...........] - ETA: 6s - ndcg_metric: 0.9343 - root_mean_squared_error: 3.2592 - loss: 2309.6582 - regularization_loss: 0.
 16/23 [===================>..........] - ETA: 5s - ndcg_metric: 0.9343 - root_mean_squared_error: 3.2517 - loss: 2276.0726 - regularization_loss: 0.
 17/23 [=====================>........] - ETA: 4s - ndcg_metric: 0.9343 - root_mean_squared_error: 3.2934 - loss: 2278.4628 - regularization_loss: 0.
 18/23 [======================>.......] - ETA: 3s - ndcg_metric: 0.9344 - root_mean_squared_error: 3.2636 - loss: 2303.9164 - regularization_loss: 0.
 19/23 [=======================>......] - ETA: 3s - ndcg_metric: 0.9343 - root_mean_squared_error: 3.2782 - loss: 2323.8223 - regularization_loss: 0.
 20/23 [=========================>....] - ETA: 2s - ndcg_metric: 0.9342 - root_mean_squared_error: 3.2664 - loss: 2392.5091 - regularization_loss: 0.
 21/23 [==========================>...] - ETA: 1s - ndcg_metric: 0.9343 - root_mean_squared_error: 3.2790 - loss: 2563.9430 - regularization_loss: 0.
 22/23 [===========================>..] - ETA: 0s - ndcg_metric: 0.9344 - root_mean_squared_error: 3.2710 - loss: 2522.7708 - regularization_loss: 0.
 23/23 [==============================] - ETA: 0s - ndcg_metric: 0.9343 - root_mean_squared_error: 3.2789 - loss: 2501.3938 - regularization_loss: 0.0000e+00 - total_loss: 2501.39385)
(Worker pid=6485, ip=172.16.0.135) 2022-05-24 17:35:55.788711: W tensorflow/core/framework/dataset.cc:768] Input of GeneratorDatasetOp::Dataset will
23/23 [==============================] - 18s 802ms/step - ndcg_metric: 0.9343 - root_mean_squared_error: 3.2789 - loss: 2481.7983 - regularization_loss: 0.0000e+00 - total_loss: 2481.7983 - val_ndcg_metric: 0.9030 - val_root_mean_squared_error: 3.4823 - val_loss: 265.6143 - val_regularization_loss: 0.0000e+00 - val_total_loss: 265.6143
Stopping orca context
```
