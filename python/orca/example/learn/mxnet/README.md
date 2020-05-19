# MXNet LeNet example

Here we demonstrate how to easily run synchronous distributed [MXNet](https://github.com/apache/incubator-mxnet) training using 
MXNetTrainer implemented in Analytics Zoo on top of [RayOnSpark](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/).
We use the LeNet model to train on MNIST dataset for handwritten digit recognition. 
See [here](https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/image/mnist.html) for the original single-node version of this example provided by MXNet.

In the distributed setting, the whole MNIST dataset is splitted into several parts and each MXNet worker takes a part for data parallel training. 
At the same time, MXNet servers are responsible for aggregating the parameters and send back to workers.

## Prepare environments
Follow steps 1 to 4 [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/#steps-to-run-rayonspark) 
to prepare your python environment.

You also need to install **MXNet** in your conda environment via pip. We have tested on MXNet 1.6.0.
```bash
pip install mxnet==1.6.0
```
If you are running on Intel Xeon scalable processors, you probably want to install the [MKLDNN](https://github.com/oneapi-src/oneDNN) version of MXNet for better performance:
```bash
pip install mxnet-mkl==1.6.0
```

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install)
for more running guidance after pip install. 

## Run on local after pip install
```
python lenet_mnist.py -n 2
```
See [here](#Options) for more configurable options for this example.

## Run on yarn cluster for yarn-client mode after pip install 
```
python lenet_mnist.py --hadoop_conf ...# path to your hadoop/yarn directory --conda_name ...# your conda name
```
 
See [here](#Options) for more configurable options for this example.

## Options
- `-n` `--num_workers` The number of MXNet workers to be launched for distributed training. Default is 2.
- `-s` `--num_servers` The number of MXNet servers to be launched for distributed training. If not specified, default to be equal to the number of workers.
- `-b` `--batch_size` The number of samples per gradient update for each worker. Default is 100.
- `-e` `--epochs` The number of epochs to train the model. Default is 10.
- `-l` `--learning_rate` The learning rate for the TextClassifier model. Default is 0.01.
- `--log_interval` The number of batches to wait before logging throughput and metrics information during the training process.

**Options for yarn only**
- `--hadoop_conf` This option is **required** when you want to run on yarn. The path to your configuration folder of hadoop.
- `--conda_name` This option is **required** when you want to run on yarn. The name of your conda environment.
- `--executor_cores` The number of executor cpu cores you want to use. Default is 4.

## Results
You can find the accuracy information from the log during the training process:
```
(pid=34395) INFO:root:Epoch[6] Batch[99]  Speed: 4513.416662 samples/sec  loss=0.893286  accuracy=0.961562
(pid=34361) INFO:root:Epoch[6] Batch[99]  Speed: 4465.811376 samples/sec  loss=0.900434  accuracy=0.966484
(pid=34395) INFO:root:Epoch[6] Batch[199]  Speed: 4877.452140 samples/sec  loss=0.875496  accuracy=0.964102
(pid=34361) INFO:root:Epoch[6] Batch[199]  Speed: 4322.318930 samples/sec  loss=0.901634  accuracy=0.965000
(pid=34395) INFO:root:[Epoch 6] time cost: 7.500912
(pid=34395) INFO:root:[Epoch 6] training: accuracy=0.964410  
(pid=34361) INFO:root:[Epoch 6] time cost: 7.522801
(pid=34361) INFO:root:[Epoch 6] training: accuracy=0.965178  
(pid=34395) INFO:root:[Epoch 6] validation: accuracy=0.982171  
(pid=34361) INFO:root:[Epoch 6] validation: accuracy=0.957131  
```
Note that the training and validation accuracy of each worker may slightly differ as the accuracy of each worker is calculated
 based on its own portion of the whole dataset.
 