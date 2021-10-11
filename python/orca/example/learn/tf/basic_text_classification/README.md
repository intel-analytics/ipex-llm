# Orca Tensorflow Text Classification example with movie reviews

We demonstrate how to easily run synchronous distributed Tensorflow training using Tensorflow Estimator of Project Orca in Analytics Zoo. This example classifies movie reviews as positive or negative using the text of the review. See [here](https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/keras/basic_text_classification.ipynb) for the original single-node version of this example provided by Tensorflow.

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
You can run this example on local mode and yarn client mode.

- Run with Spark Local mode:
```bash
python basic_text_classification.py --cluster_mode local
```

- Run with Yarn Client mode:
```bash
python basic_text_classification.py --cluster_mode yarn
```

In above commands
* `--cluster_mode` The mode of spark cluster, supporting local and yarn. Default is "local".


## Results
You can find the logs for training:
```
 DistriOptimizer$:426 - [Epoch 10 5632/14848][Iteration 272][Wall Clock 16.805771461s] Trained 512 records in 0.034776978 seconds. Throughput is 14722.383 records/second. Loss is 0.38606125. 
```
And after validation, test results will be seen like:
```
2020-12-25 15:52:43 INFO  DistriOptimizer$:111 - [Epoch 10 14848/14848][Iteration 290][Wall Clock 17.47923692s] Validate model...
2020-12-25 15:52:44 INFO  DistriOptimizer$:177 - [Epoch 10 14848/14848][Iteration 290][Wall Clock 17.47923692s] validate model throughput is 119.15809 records/second
2020-12-25 15:52:44 INFO  DistriOptimizer$:180 - [Epoch 10 14848/14848][Iteration 290][Wall Clock 17.47923692s] bigdl_metric_0 Loss is (Loss: 31.82238, count: 80, Average Loss: 0.39777976)
2020-12-25 15:52:44 INFO  DistriOptimizer$:180 - [Epoch 10 14848/14848][Iteration 290][Wall Clock 17.47923692s] bigdl_metric_1 Top1Accuracy is Accuracy(correct: 8548, count: 10000, accuracy: 0.8548)
```
