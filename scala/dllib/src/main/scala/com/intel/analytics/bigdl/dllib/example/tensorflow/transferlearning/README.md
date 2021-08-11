# Summary

Training a big model on a new domain from scratch can be very hard job. Since a lot of data in
the new domain is required to prevent over-fitting and the training process can be extremely
time-consuming due to the size of the model and training data. A very common approach to this
problem is transfer learning, which uses a pre-trained model of a similar task and fine-tunes it
for the task at hand.

In this example, we will show you how to use a pre-trained tensorflow inception-v1 model trained on
imagenet dataset to solve the flowers classification problem by fine-tune it in BigDL. As the flowers
dataset contains only thousands of images, we will treat the inception-v1 model as a feature extractor
and only train a linear model on these features.

# Preparation

## Make sure Spark, BigDL (both scala and python api) and Tensorflow are successfully install

Please refer to [BigDL](https://bigdl-project.github.io/master/), [Tensorflow](https://www.tensorflow.org/versions/r1.10/install/) for more information.

We currently support Tensorflow r1.10.

```shell
pip install tensorflow==1.10
```

## Install the TF-slim image models library

Please checkout this [page](https://github.com/tensorflow/models/tree/master/research/slim#installing-the-tf-slim-image-models-library)
to install the TF-slim image models library. And add the library to `PYTHONPATH`.

For example,

```shell
SLIM_PATH=/your/path/to/slim/model/
export PYTHONPATH=$SLIM_PATH:$PYTHONPATH
```

## Get the flowers datasets

```shell
mkdir /tmp/flowers
cd /tmp/flowers
wget http://download.tensorflow.org/data/flowers.tar.gz
tar -xzvf flowers.tar.gz
```

## Get the inception-v1 checkpoint file

```shell
mkdir /tmp/checkpoints
cd /tmp/checkpoints
wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
tar -xzvf inception_v1_2016_08_28.tar.gz
```

## Dump the pre-trained tensorflow model to a BigDL readable format

By reading directly from tensorflow's computational graph, BigDL can recover
the entire training pipeline, including the in-graph preprocessing step and the
model, and return the extracted features as an RDD. 

To apply this pipeline to a new dataset, you should made a few modifications to your original training
code as illustrated in the dump_model_example.py script:

  1. Change the original dataset to the new one of the problem at hand.
  2. We may also set the model in evaluation model, since we only treat
  it as a feature extractor and we do not want the additional stochasticity
  introduced by Dropout layer or data augmentation process.
  3. Comment out the actually training code and use BigDL `dump_model` function
  to write the model as BigDL readable format.
  
You can easily apply these steps to your own model.

For this example, you can simply run the following commands:

```shell
mkdir /tmp/tf_model_train/
python dump_model_example.py --data_split=train --dump_model_path=/tmp/tf_model_train/
mkdir /tmp/tf_model_validation/
python dump_model_example.py --data_split=validation --dump_model_path=/tmp/tf_model_validation/
```

One thing to note is that data path is hard coded in tensorflow's computational graph, so we need
to dump two pipeline, one for training and one for validation.

# Training for flowers classification

## Loading the Pipeline

Provided the pipeline definition in `model.pb`, pre-trained weights in `model.bin` and
the specified end-points, we can load the entire pipeline and return the results as an
RDD. In this case, the resulting rdd contains the features extracted from inception-v1 model
as well as their corresponding label.

```scala

    val training_graph_file = modelDir + "/model.pb"
    val training_bin_file = modelDir + "/model.bin"

    val featureOutputNode = "InceptionV1/Logits/AvgPool_0a_7x7/AvgPool"
    val labelNode = "OneHotEncoding/one_hot"

    val session = TensorflowLoader.checkpoints[Float](training_graph_file,
      training_bin_file, ByteOrder.LITTLE_ENDIAN)
      .asInstanceOf[BigDLSessionImpl[Float]]

    val rdd = session.getRDD(Seq(featureOutputNode, labelNode), sc)

```

## Train a Classifier

We will only train a linear using the features extracted by inception-v1 model.

```scala

      val model = Sequential[Float]()
      model.add(Squeeze[Float](null, batchMode = true))
      model.add(Linear[Float](1024, 5))

      val criterion = CrossEntropyCriterion[Float]()

      val optimizer = Optimizer[Float](model, trainingData, criterion, param.batchSize)

      val endWhen = Trigger.maxEpoch(param.nEpochs)
      val optim = new RMSprop[Float](learningRate = 0.001, decayRate = 0.9)

      optimizer.setEndWhen(endWhen)
      optimizer.setOptimMethod(optim)

      optimizer.optimize()
```

The `TransferLearning.scala` contains the entire code for this task.

To run this example, you can modify and execute the following command:

```shell
SPARK_HOME=...
BIGDL_HOME=...
BIGDL_VERSION=...
$SPARK_HOME/bin/spark-submit \
--master spark://... \
--driver-memory driver-memory \
--executor-memory executor-memory \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--driver-class-path $BIGDL_HOME/lib/bigdl-$BIGDL_VERSION-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.example.tensorflow.transferlearning.TransferLearning  \
$BIGDL_HOME/lib/bigdl-$BIGDL_VERSION-jar-with-dependencies.jar \
-t /tmp/tf_model_train/ -v /tmp/tf_model_validation/ \
-b batch_size -e nEpochs
```

After training, you should see something like this in the console:

```
2017-11-16 12:13:02 INFO  DistriOptimizer$:330 - [Epoch 10 3328/3320][Iteration 2080][Wall Clock 201.064860541s] Trained 16 records in 0.01422112 seconds. Throughput is 1125.0872 records/second. Loss is 0.15326343. 
2017-11-16 12:13:02 INFO  DistriOptimizer$:374 - [Epoch 10 3328/3320][Iteration 2080][Wall Clock 201.064860541s] Epoch finished. Wall clock time is 201105.290228 ms
2017-11-16 12:13:02 INFO  DistriOptimizer$:626 - [Wall Clock 201.105290228s] Validate model...
2017-11-16 12:13:02 INFO  DistriOptimizer$:668 - Top1Accuracy is Accuracy(correct: 306, count: 350, accuracy: 0.8742857142857143)
```
As we can see, training a linear classifier for 10 epochs, we can achieve a 
87.4% accuracy on the validation set.

  
  







