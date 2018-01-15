

## Scala Models

BigDL provides loads of popular models ready for use in your application. Some of them are listed blow. See all in [scala neural network models](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models). 

   * [LeNet](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/lenet): it demonstrates how to use BigDL to train and evaluate the [LeNet-5](http://yann.lecun.com/exdb/lenet/) network on MNIST data.
   * [Inception](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/inception): it demonstrates how to use BigDL to train and evaluate [Inception v1](https://arxiv.org/abs/1409.4842) and [Inception v2](https://arxiv.org/abs/1502.03167) architecture on the ImageNet data.
   * [VGG](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/vgg): it demonstrates how to use BigDL to train and evaluate a [VGG-like](http://torch.ch/blog/2015/07/30/cifar.html) network on CIFAR-10 data.
   * [ResNet](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/resnet): it demonstrates how to use BigDL to train and evaluate the [ResNet](https://arxiv.org/abs/1512.03385) architecture on CIFAR-10 data.
   * [RNN](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/rnn): it demonstrates how to use BigDL to build and train a simple recurrent neural network [(RNN) for language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf).
   * [Auto-encoder](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/autoencoder): it demonstrates how to use BigDL to build and train a basic fully-connected autoencoder using MNIST data.

---
## Scala Example

BigDL ships plenty of Scala examples to show how to use BigDL to solve real problems. Some are listed blow. See all of them in [scala deep learning examples](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example) 

   * [Text Classification](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/textclassification):
    it demonstrates how to use BigDL to build a [text classifier](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) using a simple convolutional neural network (CNN) model.
   * [Image Classification](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/imageclassification):
    it demonstrates how to load a BigDL or [Torch](http://torch.ch/) model trained on ImageNet data (e.g., [Inception](https://arxiv.org/abs/1409.4842) or [ResNet](https://arxiv.org/abs/1512.03385)),
    and then applies the loaded model to classify the contents of a set of images in Spark ML pipeline.
   * [Load Model](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/loadmodel):
    it demonstrates how to use BigDL to load a pre-trained [Torch](http://torch.ch/) or [Caffe](http://caffe.berkeleyvision.org/) model into Spark program for prediction.
   * [ML Pipline](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/MLPipeline):
    it demonstrates how to use BigDL DLClassifier to train a Logistic Regression Model. DLClassifier extends Spark Estimator and can act as a stage in a ML Pipeline.
   * [TreeLSTM Sentiment](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/treeLSTMSentiment):
    it demonstrates how to use BigDL train a model on [Stanford Treebank dataset](https://nlp.stanford.edu/sentiment/index.html) dataset using binary TreeLSTM and [Glove](https://nlp.stanford.edu/projects/glove/)
    word embedding vectors.
   * [UDF Predictor](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/udfpredictor):
    it demonstrates how to load BigDL model as UDF to perform predictions in Spark SQL/Dataframes.
