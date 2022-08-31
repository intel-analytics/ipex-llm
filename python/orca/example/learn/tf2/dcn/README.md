# Running Orca TF2 DCN example


## Environment

We recommend conda to set up your environment. You can install a conda distribution from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
if you haven't already.

```bash
conda create -n bigdl python==3.7
conda activate bigdl
pip install tensorflow pandas pyarrow pillow numpy
```

Then download and install latest nightly-build BigDL Orca.

```bash
pip install --pre --upgrade bigdl-orca[ray]
pip install --pre --upgrade bigdl-friesian
```

## Training Data

Download MovieLens 1M Dataset [here](https://grouplens.org/datasets/movielens/1m/) and unzip the ml-1m.zip file.


## Preprocess

Replace data_dir in to_parquet.py L19 with the /path/to/ml-1m/folder
```bash
python to_parquet.py
```
You will get total.parquet file under the ml-1m folder.


## Running example

### Train

#### Local
```bash
python dcn_parquet.py --data_dir /path/to/total.parquet
```
Result:
```bash
(Worker pid=28939) Epoch 8/8
 1/10 [==>...........................] - ETA: 0s - RMSE: 0.8372 - loss: 0.7009 - regularization_loss: 0.0000e+00 - total_loss: 0.7009
 3/10 [========>.....................] - ETA: 0s - RMSE: 0.8228 - loss: 0.6771 - regularization_loss: 0.0000e+00 - total_loss: 0.6771
 4/10 [===========>..................] - ETA: 0s - RMSE: 0.8237 - loss: 0.6786 - regularization_loss: 0.0000e+00 - total_loss: 0.6786
 6/10 [=================>............] - ETA: 0s - RMSE: 0.8270 - loss: 0.6840 - regularization_loss: 0.0000e+00 - total_loss: 0.6840
 7/10 [====================>.........] - ETA: 0s - RMSE: 0.8276 - loss: 0.6850 - regularization_loss: 0.0000e+00 - total_loss: 0.6850
 9/10 [==========================>...] - ETA: 0s - RMSE: 0.8249 - loss: 0.6804 - regularization_loss: 0.0000e+00 - total_loss: 0.6804
10/10 [==============================] - ETA: 0s - RMSE: 0.8244 - loss: 0.6797 - regularization_loss: 0.0000e+00 - total_loss: 0.6797
10/10 [==============================] - 1s 122ms/step - RMSE: 0.8244 - loss: 0.6791 - regularization_loss: 0.0000e+00 - total_loss: 0.6791 - val_RMSE: 0.9650 - val_loss: 0.9424 - val_regularization_loss: 0.0000e+00 - val_total_loss: 0.9424
[Stage 69:>                                                         (0 + 2) / 2]
[Stage 72:>                                                         (0 + 2) / 2]2022-04-28 16:50:10,247	INFO worker.py:843 -- Connecting to existing Ray cluster at address: 127.0.0.1:6379
2022-04-28 16:50:10,247	INFO worker.py:843 -- Connecting to existing Ray cluster at address: 127.0.0.1:6379
1/3 [=========>....................] - ETA: 0s - RMSE: 0.9712 - loss: 0.9432 - regularization_loss: 0.0000e+00 - total_loss: 0.9432
3/3 [==============================] - 0s 39ms/step - RMSE: 0.9650 - loss: 0.9340 - regularization_loss: 0.0000e+00 - total_loss: 0.9340
```

#### YARN
```bash
python dcn_parquet.py --data_dir hdfs://path/to/total.parquet --cluster_mode yarn --executor_cores 8 --executor_memory 10g --num_executor 2
```

Result:
```bash
(Worker pid=100385, ip=172.16.0.161) Epoch 8/8
 1/10 [==>...........................] - ETA: 0s - RMSE: 0.8499 - loss: 0.7134 - regularization_loss: 0.0000e+00 
 2/10 [=====>........................] - ETA: 0s - RMSE: 0.8564 - loss: 0.7415 - regularization_loss: 0.0000e+00 
 4/10 [===========>..................] - ETA: 0s - RMSE: 0.8600 - loss: 0.7477 - regularization_loss: 0.0000e+00 
 5/10 [==============>...............] - ETA: 0s - RMSE: 0.8646 - loss: 0.7548 - regularization_loss: 0.0000e+00 
 6/10 [=================>............] - ETA: 0s - RMSE: 0.8636 - loss: 0.7535 - regularization_loss: 0.0000e+00 
 8/10 [=======================>......] - ETA: 0s - RMSE: 0.8626 - loss: 0.7488 - regularization_loss: 0.0000e+00 
 9/10 [==========================>...] - ETA: 0s - RMSE: 0.8620 - loss: 0.7468 - regularization_loss: 0.0000e+00
 10/10 [==============================] - ETA: 0s - RMSE: 0.8630 - loss: 0.7469 - regularization_loss: 0.0000e+00
 10/10 [==============================] - 1s 108ms/step - RMSE: 0.8630 - loss: 0.7469 - regularization_loss: 0.0000e+00 - total_loss: 0.7469 - val_RMSE: 0.9492 - val_loss: 0.8921 - val_regularization_loss: 0.0000e+00 - val_total_loss: 0.8921
(LocalStore pid=106877, ip=172.16.0.161) 2022-04-28 19:42:36.587430: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: python_env/lib:python_env/lib/python3.7/lib-dynload::/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib/hadoop/lib/native
(LocalStore pid=106877, ip=172.16.0.161) 2022-04-28 19:42:36.587483: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
1/3 [=========>....................] - ETA: 0s - RMSE: 0.9399 - loss: 0.8947 - regularization_loss: 0.0000e+00 -
3/3 [==============================] - 0s 32ms/step - RMSE: 0.9492 - loss: 0.9012 - regularization_loss: 0.0000e+00 - total_loss: 0.901272.16.0.161)
```
