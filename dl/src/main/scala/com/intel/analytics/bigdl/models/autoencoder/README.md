# Autoencoder example on MNIST

Autoencoder is used for unsupervised learning, and this model is the basic fully-connected autoencoder.

## Data files needed:

- train-images-idx3-ubyte
- train-labels-idx1-ubyte (the labels file is not actually used)

You can download them at [MNIST](http://yann.lecun.com/exdb/mnist/) and put them at ./ or the path you set with `-f`.

## Build:

mvn clean package -DskipTests

## Train on Spark:

`spark-submit --class com.intel.analytics.bigdl.models.autoencoder.Train bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --node 8 --core 10 -b 400 --env spark`

## Train on Local:

`java -cp bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.autoencoder.Train --core 1 -node 1 --env local`
