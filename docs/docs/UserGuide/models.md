---
# **Models** 
---

BigDL provides many popular [neural network models](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models) and [deep learning examples](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example) for Apache Spark, including: 

## **Models**
   * [LeNet](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/lenet): it demonstrates how to use BigDL to train and evaluate the [LeNet-5](http://yann.lecun.com/exdb/lenet/) network on MNIST data.
   * [Inception](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/inception): it demonstrates how to use BigDL to train and evaluate [Inception v1](https://arxiv.org/abs/1409.4842) and [Inception v2](https://arxiv.org/abs/1502.03167) architecture on the ImageNet data.
   * [VGG](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/vgg): it demonstrates how to use BigDL to train and evaluate a [VGG-like](http://torch.ch/blog/2015/07/30/cifar.html) network on CIFAR-10 data.
   * [ResNet](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/resnet): it demonstrates how to use BigDL to train and evaluate the [ResNet](https://arxiv.org/abs/1512.03385) architecture on CIFAR-10 data.
   * [RNN](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/rnn): it demonstrates how to use BigDL to build and train a simple recurrent neural network [(RNN) for language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf).
   * [Auto-encoder](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/autoencoder): it demonstrates how to use BigDL to build and train a basic fully-connected autoencoder using MNIST data.

## **Examples**

   * [text_classification](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/textclassification): it demonstrates how to use BigDL to build a [text classifier](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) using a simple convolutional neural network (CNN) model.
   * [image_classification](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/imageclassification): it demonstrates how to load a BigDL or [Torch](http://torch.ch/) model trained on ImageNet data (e.g., [Inception](https://arxiv.org/abs/1409.4842) or [ResNet](https://arxiv.org/abs/1512.03385)), and then applies the loaded model to classify the contents of a set of images in Spark ML pipeline.
   * [load_model](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/loadmodel): it demonstrates how to use BigDL to load a pre-trained [Torch](http://torch.ch/) or [Caffe](http://caffe.berkeleyvision.org/) model into Spark program for prediction.

## **Python Examples**
   * [LeNet](https://github.com/intel-analytics/BigDL/tree/master/pyspark/dl/models/lenet): it demonstrates how to use BigDL Python APIs to train and evaluate the [LeNet-5](http://yann.lecun.com/exdb/lenet/) network on MNIST data.
   * [Text Classifier](https://github.com/intel-analytics/BigDL/tree/master/pyspark/dl/models/textclassifier):  it demonstrates how to use BigDL Python APIs to build a text classifier using a simple [convolutional neural network (CNN) model(https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)] or a simple LSTM/GRU model.
   * [Jupyter tutorial](https://github.com/intel-analytics/BigDL/blob/branch-0.1/pyspark/dl/example/tutorial/simple_text_classification/text_classfication.ipynb): it contains a tutorial for using BigDL Python APIs in Jupyter notebooks (together with TensorBoard support) for interactive data explorations and visualizations.