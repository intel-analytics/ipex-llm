# Orca Ray Pong example

We demonstrate how to easily run [pong](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5) example
provided by [Ray](https://github.com/ray-project/ray) using Analytics Zoo API. See [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/) for more details for RayOnSpark support in Analytics Zoo.

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster(yarn-client mode only).
```
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install gym gym[atari]
pip install analytics-zoo[ray]
```

## Run example
You can run this example on local mode and yarn client mode. 

- Run with Spark Local mode:
```bash
python rl_pong.py
```

- Run with Yarn Client mode:
```bash
python rl_pong.py --cluster_mode yarn
```

In above commands
* `--cluster_mode` The mode of spark cluster, supporting local and yarn. Default is "local".
- `--object_store_memory`The store memory you need to use on local. Default is 4g.
- `--driver_cores` The number of driver's or local's cpu cores you want to use. Default is 8.
- `--iterations` The number of iterations to train the model. Default is -1, training will not terminate.
- `--batch_size` The number of roll-outs to do per batch. Default is 10.

**Options for yarn only**
- `--slave_num` The number of slave nodes you want to to use. Default is 2.
- `--executor_cores` The number of slave(executor)'s cpu cores you want to use. Default is 8.
- `--executor_memory` The size of slave(executor)'s memory you want to use. Default is 10g.
- `--driver_memory` The size of driver's memory you want to use. Default is 2g
- `--extra_executor_memory_for_ray` The size of slave(executor)'s extra memory to store data. Default is 20g.


## Results
You can find the logs for training:
```
Batch 3 computed 10 rollouts in 3.907017707824707 seconds, running mean is -20.851752130222128
```
