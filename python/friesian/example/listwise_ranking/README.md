# Running Friesian listwise example

This example is based on Tensorflow Recommenders example [Listwise ranking](https://github.com/tensorflow/recommenders/blob/main/docs/examples/listwise_ranking.ipynb)

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

### Sample listwise ranking

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

### Pad listwise ranking

This example pads all user's movie ratings to the same length with mask_tokens, and only uses valid ratings to compute the loss.
Losses in this example should support ragged tensor, like:
```python
tfr.keras.losses.ListMLELoss(ragged=True)
```

#### Local
```bash
python listwise_pad.py --data_dir /path/to/ml-1m
```
Result:
```bash
(Worker pid=5522) Epoch 16/16
 1/24 [>.............................] - ETA: 13s - ndcg_metric: 0.9222 - root_mean_squared_error: 4.7729 - loss: 3816.8352 - regularization_loss: 0.0000e+00 - total_loss: 3816.8352
 2/24 [=>............................] - ETA: 12s - ndcg_metric: 0.9196 - root_mean_squared_error: 4.8949 - loss: 6329.1027 - regularization_loss: 0.0000e+00 - total_loss: 6329.1027
 3/24 [==>...........................] - ETA: 12s - ndcg_metric: 0.9208 - root_mean_squared_error: 4.9284 - loss: 6106.6015 - regularization_loss: 0.0000e+00 - total_loss: 6106.6015
 4/24 [====>.........................] - ETA: 11s - ndcg_metric: 0.9196 - root_mean_squared_error: 4.9295 - loss: 6032.5925 - regularization_loss: 0.0000e+00 - total_loss: 6032.5925
 5/24 [=====>........................] - ETA: 10s - ndcg_metric: 0.9202 - root_mean_squared_error: 4.9425 - loss: 5966.7832 - regularization_loss: 0.0000e+00 - total_loss: 5966.7832
 6/24 [======>.......................] - ETA: 10s - ndcg_metric: 0.9204 - root_mean_squared_error: 4.9532 - loss: 5639.7241 - regularization_loss: 0.0000e+00 - total_loss: 5639.7241
 7/24 [=======>......................] - ETA: 9s - ndcg_metric: 0.9205 - root_mean_squared_error: 4.9922 - loss: 6549.3948 - regularization_loss: 0.0000e+00 - total_loss: 6549.3948 
 8/24 [=========>....................] - ETA: 9s - ndcg_metric: 0.9203 - root_mean_squared_error: 4.9901 - loss: 6639.0820 - regularization_loss: 0.0000e+00 - total_loss: 6639.0820
 9/24 [==========>...................] - ETA: 8s - ndcg_metric: 0.9207 - root_mean_squared_error: 5.0028 - loss: 6883.1433 - regularization_loss: 0.0000e+00 - total_loss: 6883.1433
10/24 [===========>..................] - ETA: 8s - ndcg_metric: 0.9213 - root_mean_squared_error: 5.0188 - loss: 6652.6776 - regularization_loss: 0.0000e+00 - total_loss: 6652.6776
11/24 [============>.................] - ETA: 7s - ndcg_metric: 0.9217 - root_mean_squared_error: 5.0115 - loss: 6499.7629 - regularization_loss: 0.0000e+00 - total_loss: 6499.7629
12/24 [==============>...............] - ETA: 6s - ndcg_metric: 0.9214 - root_mean_squared_error: 5.0063 - loss: 6292.0559 - regularization_loss: 0.0000e+00 - total_loss: 6292.0559
13/24 [===============>..............] - ETA: 6s - ndcg_metric: 0.9216 - root_mean_squared_error: 5.0199 - loss: 6735.6548 - regularization_loss: 0.0000e+00 - total_loss: 6735.6548
14/24 [================>.............] - ETA: 5s - ndcg_metric: 0.9216 - root_mean_squared_error: 5.0202 - loss: 6652.8493 - regularization_loss: 0.0000e+00 - total_loss: 6652.8493
15/24 [=================>............] - ETA: 5s - ndcg_metric: 0.9215 - root_mean_squared_error: 5.0084 - loss: 6780.2701 - regularization_loss: 0.0000e+00 - total_loss: 6780.2701
16/24 [===================>..........] - ETA: 4s - ndcg_metric: 0.9211 - root_mean_squared_error: 4.9963 - loss: 6615.7784 - regularization_loss: 0.0000e+00 - total_loss: 6615.7784
17/24 [====================>.........] - ETA: 4s - ndcg_metric: 0.9209 - root_mean_squared_error: 4.9872 - loss: 6469.5760 - regularization_loss: 0.0000e+00 - total_loss: 6469.5760
18/24 [=====================>........] - ETA: 3s - ndcg_metric: 0.9210 - root_mean_squared_error: 4.9910 - loss: 6462.2808 - regularization_loss: 0.0000e+00 - total_loss: 6462.2808
19/24 [======================>.......] - ETA: 2s - ndcg_metric: 0.9212 - root_mean_squared_error: 4.9978 - loss: 6437.4302 - regularization_loss: 0.0000e+00 - total_loss: 6437.4302
20/24 [========================>.....] - ETA: 2s - ndcg_metric: 0.9214 - root_mean_squared_error: 4.9996 - loss: 6485.3018 - regularization_loss: 0.0000e+00 - total_loss: 6485.3018
21/24 [=========================>....] - ETA: 1s - ndcg_metric: 0.9214 - root_mean_squared_error: 4.9988 - loss: 6369.8582 - regularization_loss: 0.0000e+00 - total_loss: 6369.8582
22/24 [==========================>...] - ETA: 1s - ndcg_metric: 0.9215 - root_mean_squared_error: 5.0041 - loss: 6323.3920 - regularization_loss: 0.0000e+00 - total_loss: 6323.3920
23/24 [===========================>..] - ETA: 0s - ndcg_metric: 0.9214 - root_mean_squared_error: 4.9998 - loss: 6228.8715 - regularization_loss: 0.0000e+00 - total_loss: 6228.8715
24/24 [==============================] - ETA: 0s - ndcg_metric: 0.9217 - root_mean_squared_error: 5.0059 - loss: 6276.6422 - regularization_loss: 0.0000e+00 - total_loss: 6276.6422
```

#### YARN
```bash
python listwise_pad.py --data_dir hdfs://path/to/ml-1m --cluster_mode yarn --executor_cores 8 --executor_memory 50g --num_executor 2
```

Result:
```bash
(Worker pid=53641, ip=172.16.0.116) Epoch 15/16
24/24 [==============================] - 16s 658ms/step - ndcg_metric: 0.9188 - root_mean_squared_error: 3.0038 - loss: 2398.3484 - regularization_loss: 0.0000e+00 - total_loss: 2398.3484 - val_ndcg_metric: 0.9099 - val_root_mean_squared_error: 3.0186 - val_loss: 542.2716 - val_regularization_loss: 0.0000e+00 - val_total_loss: 542.2716
(Worker pid=53641, ip=172.16.0.116) Epoch 16/16
24/24 [==============================] - ETA: 0s - ndcg_metric: 0.9208 - root_mean_squared_error: 3.0909 - loss: 2615.6834 - regularization_loss: 0.0000e+00 - total_loss: 2615.6834
(Worker pid=53642, ip=172.16.0.116) 2022-06-02 17:11:05.033809: W tensorflow/core/framework/dataset.cc:768] Input of GeneratorDatasetOp::Dataset will not be optimized because the dataset does not implement the AsGraphDefInternal() method needed to apply optimizations.
24/24 [==============================] - 16s 668ms/step - ndcg_metric: 0.9208 - root_mean_squared_error: 3.0909 - loss: 2625.1290 - regularization_loss: 0.0000e+00 - total_loss: 2625.1290 - val_ndcg_metric: 0.9096 - val_root_mean_squared_error: 3.0197 - val_loss: 542.4033 - val_regularization_loss: 0.0000e+00 - val_total_loss: 542.4033
(LocalStore pid=86721, ip=172.16.0.116) 2022-06-02 17:11:12.550187: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: python_env/lib:python_env/lib/python3.7/lib-dynload::/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib/hadoop/lib/native
(LocalStore pid=86721, ip=172.16.0.116) 2022-06-02 17:11:12.550235: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
 3/24 [==>...........................] - ETA: 0s - ndcg_metric: 0.9088 - root_mean_squared_error: 2.9492 - loss: 551.2205 - regularization_loss 
 5/24 [=====>........................] - ETA: 0s - ndcg_metric: 0.9092 - root_mean_squared_error: 3.0018 - loss: 546.1960 - regularization_loss 
 7/24 [=======>......................] - ETA: 0s - ndcg_metric: 0.9074 - root_mean_squared_error: 3.0239 - loss: 584.6721 - regularization_loss 
 9/24 [==========>...................] - ETA: 0s - ndcg_metric: 0.9079 - root_mean_squared_error: 3.0398 - loss: 632.1029 - regularization_loss
 11/24 [============>.................] - ETA: 0s - ndcg_metric: 0.9078 - root_mean_squared_error: 3.0286 - loss: 620.9329 - regularization_loss
 13/24 [===============>..............] - ETA: 0s - ndcg_metric: 0.9083 - root_mean_squared_error: 3.0241 - loss: 633.5845 - regularization_loss
 17/24 [====================>.........] - ETA: 0s - ndcg_metric: 0.9088 - root_mean_squared_error: 3.0236 - loss: 635.6947 - regularization_loss
 19/24 [======================>.......] - ETA: 0s - ndcg_metric: 0.9095 - root_mean_squared_error: 3.0207 - loss: 620.3506 - regularization_loss
 21/24 [=========================>....] - ETA: 0s - ndcg_metric: 0.9096 - root_mean_squared_error: 3.0150 - loss: 606.1091 - regularization_loss
 23/24 [===========================>..] - ETA: 0s - ndcg_metric: 0.9096 - root_mean_squared_error: 3.0227 - loss: 650.2150 - regularization_loss: 0.0000e+00 - total_loss: 650.2150
Stopping orca context
```
