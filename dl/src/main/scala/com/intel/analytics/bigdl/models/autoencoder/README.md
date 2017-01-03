# Autoencoder example on MNIST

Autoencoder[<a href="#Bengio09">Bengio09</a>] is an unsupervised learning model, and this model is a
basic unit of Stacked Autoencoders.

## Data Used:
To train the autoencoder, you need the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

You need to download:

- train-images-idx3-ubyte.gz
- train-labels-idx1-ubyte.gz (the labels file is not actually used)

then unzip them, you can get:
- train-images-idx3-ubyte
- train-labels-idx1-ubyte

## Train on Spark:
Enter the following commands to run the model on Spark:
```{r, engine='sh'}
$ ./dist/bin/bigdl.sh -- spark-submit \
$ --class com.intel.analytics.bigdl.models.autoencoder.Train \
$ ./dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
$ --node 8 --core 10 -b 400 --env spark -f $DATA_FOLDER
```
where `$DATA_FOLDER` is the directory containing the MNIST training data, whose default value is "./ ".

## Train on Local:
Enter the following commands to run the model on local:
```{r, engine='sh'}
$ ./dist/bin/bigdl.sh -- java -cp \
$ ./dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
$ com.intel.analytics.bigdl.models.autoencoder.Train \
$ --core 1 --node 1 --env local -f $DATA_FOLDER
```
where `$DATA_FOLDER` is the directory containing the MNIST training data, whose default value is "./ ".

## Model Brief Introduction
Auto-encoder has the same dimension for both input and output and
takes an unlabeled training examples in data set and encodes it to the hidden layer by linear
combination with weight matrix and then through a non-linear activation function.

After that the hidden layer representation will be reconstructed to the output layer through a decoding function, in which the output has a same shape as the input. The aim of the model is to optimize the weight matrices,
so that the reconstruction error between input and output can be minimized. It can be seen that the Autoencoder
can be viewed as an unsupervised learning process of encoding-decoding: the encoder encodes the input through
multi-layer encoder and then the decoder will decode it back with the lowest error[<a href="#Hinton06">Hinton06</a>].

See [Implementation Details](#implementation-details) for more information on implementation.


## Implementation Details
For our implementation, ReLU is used as activation function and the mean square error as the loss function.

#### Reference
<a name="Bengio09">[Bengio09]</a> Yoshua Bengio. Learning deep architectures for AI. Foundations and Trends in Machine Learning, 2(1):1–127, 2009. Also published as a book. Now Publishers, 2009.

<a name="Hinton06">[Hinton06]</a> Geoffrey E Hinton and Ruslan R Salakhutdinov. Reducing the dimensionality of data with neural networks. Science, 313(5786):504–507, 2006