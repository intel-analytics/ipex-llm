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
python basic_ranking.py --data_dir hdfs://path/to/ml-1m --cluster_mode yarn --executor_cores 8 --executor_memory 10g --num_executor 2
```

Result:
```bash
(Worker pid=16455, ip=172.16.0.110) Epoch 1/3
(Worker pid=16455, ip=172.16.0.110) WARNING:tensorflow:AutoGraph could not transform <bound method SampleRankingModel.call of <__main__.SampleRankingModel object at 0x7f65a058d250>> and will run it as-is.
(Worker pid=16455, ip=172.16.0.110) Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
(Worker pid=16455, ip=172.16.0.110) Cause: Unknown node type <gast.gast.Import object at 0x7f54882e1890>
(Worker pid=16455, ip=172.16.0.110) To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
(Worker pid=16456, ip=172.16.0.110) WARNING:tensorflow:AutoGraph could not transform <bound method SampleRankingModel.call of <__main__.SampleRankingModel object at 0x7fa94474b250>> and will run it as-is.
(Worker pid=16456, ip=172.16.0.110) Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
(Worker pid=16456, ip=172.16.0.110) Cause: Unknown node type <gast.gast.Import object at 0x7f98244418d0>
(Worker pid=16456, ip=172.16.0.110) To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
10/10 [==============================] - 6s 77ms/step - loss: 4.5389 - root_mean_squared_error: 2.1305
(Worker pid=16455, ip=172.16.0.110) Epoch 2/3
10/10 [==============================] - 1s 70ms/step - loss: 1.4701 - root_mean_squared_error: 1.2125
(Worker pid=16455, ip=172.16.0.110) Epoch 3/3
10/10 [==============================] - 0s 34ms/step - loss: 1.2625 - root_mean_squared_error: 1.1236
(LocalStore pid=19915, ip=172.16.0.110) 2022-06-22 16:44:45.683035: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: python_env/lib:python_env/lib/python3.7/lib-dynload::/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib/hadoop/lib/native
(LocalStore pid=19915, ip=172.16.0.110) 2022-06-22 16:44:45.683090: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
3/5 [=================>............] - ETA: 0s - loss: 1.2578 - root_mean_squared_error: 1.1215
Stopping orca context=172.16.0.110)
Try to unpersist an uncached rdd
Try to unpersist an uncached rdd
```
