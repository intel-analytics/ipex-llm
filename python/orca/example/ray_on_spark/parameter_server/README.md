# Orca Ray Parameter server example

We demonstrate how to easily run [Ray](https://github.com/ray-project/ray) examples:
[async_parameter_server](https://github.com/ray-project/ray/blob/master/doc/examples/parameter_server/async_parameter_server.py)
and [sync_parameter_server](https://github.com/ray-project/ray/blob/master/doc/examples/parameter_server/sync_parameter_server.py).
See [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/) for more details for RayOnSpark support in Analytics Zoo.

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster(yarn-client mode only).
```
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install tensorflow
pip install analytics-zoo[ray]
```

## Run example
You can run this example on local mode and yarn client mode. 

- Run with Spark Local mode:
```bash
python async_parameter_server.py --iterations 20 --num_workers 2
python sync_parameter_server --iterations 20 --num_workers 2
```

- Run with Yarn Client mode:
You should download [MNIST](http://yann.lecun.com/exdb/mnist/) dataset first, and prepare package file as the flollowing steps:
```bash
# prepare dataset
zip MNIST_data.zip train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz

# run the example
python async_parameter_server.py --iterations 20 --num_workers 2 --cluster_mode yarn
python sync_parameter_server --iterations 20 --num_workers 2 --cluster_mode yarn
```

In above commands
* `--cluster_mode` The mode of spark cluster, supporting local and yarn. Default is "local".
- `--object_store_memory`The store memory you need to use on local. Default is 4g.
- `--driver_cores` The number of driver's or local's cpu cores you want to use. Default is 8.
- `--iterations` The number of iterations to train the model. Default is -1, training will not terminate.
- `--batch_size` The number of roll-outs to do per batch. Default is 10.

**Options for yarn only**
- `--num_workers` The number of slave nodes you want to to use. Default is 2.
- `--executor_cores` The number of slave(executor)'s cpu cores you want to use. Default is 8.
- `--executor_memory` The size of slave(executor)'s memory you want to use. Default is 10g.
- `--driver_memory` The size of driver's memory you want to use. Default is 2g
- `--extra_executor_memory_for_ray` The size of slave(executor)'s extra memory to store data. Default is 20g.


## Results
You can find the logs for training:
```
-----Iteration1------
Iteration 1: accuracy is 0.10300000011920929
```
