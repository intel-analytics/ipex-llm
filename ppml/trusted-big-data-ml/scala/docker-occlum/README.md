# Java Spark Occlum

## Spark 2.4.6 local test

Configure environment variables in `Dockerfile` and `build-docker-image.sh`.

Build the docker image:
``` bash
bash build-docker-image.sh
```

To train a model with PPML in Analytics Zoo and BigDL, you need to prepare the data first. The Docker image is taking lenet and MNIST as example.

You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the files and put them in a new directory `data`. There are four files in total. `train-images-idx3-ubyte` contains train images; `train-labels-idx1-ubyte` is the train label file; `t10k-images-idx3-ubyte` has validation images; `t10k-labels-idx1-ubyte` contains validation labels.

To run Spark pi example, start the docker container with:
``` bash
bash start-occlum-spark.sh test
```
To run BigDL example, start the docker container with:
``` bash
bash start-occlum-spark.sh bigdl
```
The examples are run in the docker container. Attach it and see the results.

