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

## Get the JAR
You can build one by refer to the
[Build Page](https://bigdl-project.github.io/master/#ScalaUserGuide/install-build-src/) from the source code.

## Train on Spark:
Spark local mode, example command:
```{r, engine='sh'}
spark-submit --master local[physical_core_number]\
  --class com.intel.analytics.bigdl.models.autoencoder.Train \
  ./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
  -b batch_size -f $DATA_FOLDER
```
Spark standalone mode, example command:
```{r, engine='sh'}
spark-submit --master spark://... \
  --executor-cores cores_per_executor \
  --total-executor-cores total_cores_for_the_job \
  --class com.intel.analytics.bigdl.models.autoencoder.Train \
  ./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
  -b batch_size -f $DATA_FOLDER
```
Spark yarn mode, example command:
```{r, engine='sh'}
spark-submit --master yarn --deploy-mode client\
  --executor-cores cores_per_executor \
  --num-executors executors_number \
  --class com.intel.analytics.bigdl.models.autoencoder.Train \
  ./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
  -b batch_size -f $DATA_FOLDER
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
