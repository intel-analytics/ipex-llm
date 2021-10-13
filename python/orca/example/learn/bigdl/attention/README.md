# Orca Bigdl Multi-head Attention Sentiment Analysis example

We demonstrate how to easily run synchronous distributed Bigdl training using Bigdl Estimator of Project Orca in Analytics Zoo. This example introduces the Transformer network architecture based solely on attention mechanisms, more details at paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/attention) for the original version of this example provided by analytics-zoo.

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster(yarn-client mode only).
```
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install tensorflow==1.15
pip install --pre --upgrade analytics-zoo
```

## Prepare Dataset
The IMDB dataset comes packaged with TensorFlow. It has already been preprocessed such that the reviews (sequences of words) have been converted to sequences of integers, where each integer represents a specific word in a dictionary.


## Run example
You can run this example on local mode and yarn client mode. Note that this example requires at least 10G of free memory, please check your hardware.

- Run with Spark Local mode:
```bash
python transformer.py --cluster_mode local
```

- Run with Yarn Client mode:
```bash
python transformer.py --cluster_mode yarn
```

In above commands
* `--cluster_mode` The mode of spark cluster, supporting local and yarn. Default is "local".


## Results
You can find the logs for training:
```
2021-01-27 13:41:39 INFO  DistriOptimizer$:427 - [Epoch 1 25088/25000][Iteration 196][Wall Clock 1161.351888668s] Trained 128.0 records in 5.173932028 seconds. Throughput is 24.739405 records/second. Loss is 0.151119.
```
And after validation, test results will be seen like:
```
2021-01-28 01:50:24 INFO  DistriOptimizer$:1759 - Top1Accuracy is Accuracy(correct: 17086, count: 25000, accuracy: 0.68344)
```
