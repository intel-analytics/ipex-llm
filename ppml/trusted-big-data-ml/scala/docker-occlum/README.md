# Trusted Big Data ML with Occlum


## Prerequisites

Pull image from dockerhub.

```bash
docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:0.14.0-SNAPSHOT
```

Also, you can build image with `build-docker-image.sh`. Configure environment variables in `Dockerfile` and `build-docker-image.sh`.

Build the docker image:

``` bash
bash build-docker-image.sh
```

## Spark 3.1.2 Pi example

To run Spark Pi example, start the docker container with:

``` bash
bash start-spark-local.sh test
```

You can see Pi result in logs (`docker attach logs -f containerID`)

```bash
Pi is roughly 3.1436957184785923
```

## BigDL Lenet Mnist Example

To train a model with PPML in BigDL, you need to prepare the data first. You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). There are four files in total. `train-images-idx3-ubyte` contains train images; `train-labels-idx1-ubyte` is the train label file; `t10k-images-idx3-ubyte` has validation images; `t10k-labels-idx1-ubyte` contains validation labels. Unzip all the files and put them in a new directory `data`.

By default, `data` dir will be mounted to `/opt/data`into container. You can change data path in `start-spark-local.sh`.


To run BigDL Lenet Mnist example, start the docker container with:

``` bash
bash start-spark-local.sh bigdl
```

The examples are run in the docker container. Attach it and see the results (`docker attach logs -f containerID`).

## BigDL Resnet Cifar-10 Example

Download the Cifar-10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset contains two sub-directories, namely, `train` and `val`. Put all the files and put them in a new directory `data`.

To run BigDL ResNet CIFAR-10 example, start the docker container with:

``` bash
bash start-spark-local.sh cifar
```

The examples are run in the docker container. Attach it and see the results (`docker attach logs -f containerID`).
