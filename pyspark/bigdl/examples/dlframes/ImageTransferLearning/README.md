# Summary

Python demo of transfer Learning based on Spark DataFrame (Dataset). 

BigDL provides the DataFrame-based API for image reading, preprocessing, model training and inference. The related
classes followed the typical estimator/transformer pattern of Spark ML and can be used in a standard Spark ML pipeline.

In this example, we will show you how to use a pre-trained inception-v1 model trained on
imagenet dataset to solve the dogs-vs-cats classification problem by transfer learning with BigDL.
For transfer learning, we will treat the inception-v1 model as a feature extractor and only
train a linear model on these features.

# Preparation

## Make sure Spark, BigDL are successfully installed

Please refer to [BigDL](https://bigdl-project.github.io/master/) for installation

## Get the dogs-vs-cats datasets

Download the training dataset from https://www.kaggle.com/c/dogs-vs-cats and extract it.

## Get the pre-trained Inception-V1 model

Download the pre-trained Inception-V1 model from [BigDL Zoo](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_inception-v1_imagenet_0.4.0.model)

Alternatively, user may also download pre-trained caffe/Tensorflow/keras model. Please refer to
programming guide in [BigDL](https://bigdl-project.github.io/) 

# Training for flowers classification

ImageTransferLearningExample.py takes 2 parameters:
1. Path to the pre-trained models. (E.g. path/to/model/bigdl_inception-v1_imagenet_0.4.0.model)
2. Path to the folder of the training images. (E.g. path/to/data/dogs-vs-cats/train)

User may submit ImageInferenceExample.py via spark-submit.
E.g.
```
BigDL/dist/bin/spark-submit-with-bigdl.sh --master local[4] path/BigDL/pyspark/bigdl/examples/dlframes/ImageTransferLearning/ImageTransferLearningExample.py path/to/model/bigdl_inception-v1_imagenet_0.4.0.model path/to/data/dogs-vs-cats/train
```

or run the script in Jupyter notebook or Pyspark and manually set parameters

After training, you should see something like this in the console:

```
+--------------------+------------+----+-----+--------------------+----------+
|               image|        name|  id|label|            features|prediction|
+--------------------+------------+----+-----+--------------------+----------+
|[file:/home/yuhao...|dog.1160.jpg|1160|  2.0|[3.94537119063898...|       2.0|
|[file:/home/yuhao...|cat.1001.jpg|1001|  1.0|[2.54570932156639...|       1.0|
|[file:/home/yuhao...|cat.1063.jpg|1063|  1.0|[4.04352831537835...|       1.0|
|[file:/home/yuhao...|cat.1087.jpg|1087|  1.0|[8.14869622445257...|       1.0|
|[file:/home/yuhao...|dog.1053.jpg|1053|  2.0|[3.28330497723072...|       2.0|
|[file:/home/yuhao...|cat.1020.jpg|1020|  1.0|[4.32902561442460...|       1.0|
|[file:/home/yuhao...|dog.1166.jpg|1166|  2.0|[4.99538046483394...|       2.0|
|[file:/home/yuhao...|dog.1192.jpg|1192|  2.0|[9.30676947064057...|       2.0|
|[file:/home/yuhao...|dog.1199.jpg|1199|  2.0|[7.06547325535211...|       2.0|
|[file:/home/yuhao...|dog.1129.jpg|1129|  2.0|[8.93531591827923...|       2.0|
|[file:/home/yuhao...|dog.1158.jpg|1158|  2.0|[1.99210080609191...|       2.0|
|[file:/home/yuhao...|cat.1143.jpg|1143|  1.0|[1.40288739203242...|       1.0|
|[file:/home/yuhao...|dog.1096.jpg|1096|  2.0|[4.43188264398486...|       2.0|
|[file:/home/yuhao...|dog.1086.jpg|1086|  2.0|[2.48919695877702...|       1.0|
|[file:/home/yuhao...|dog.1061.jpg|1061|  2.0|[3.77082142222207...|       2.0|
|[file:/home/yuhao...|cat.1038.jpg|1038|  1.0|[2.61234094978135...|       1.0|
|[file:/home/yuhao...|cat.1123.jpg|1123|  1.0|[4.18379180189276...|       2.0|
|[file:/home/yuhao...|dog.1025.jpg|1025|  2.0|[2.42323076236061...|       2.0|
|[file:/home/yuhao...|cat.1114.jpg|1114|  1.0|[4.40081894339527...|       1.0|
|[file:/home/yuhao...|cat.1127.jpg|1127|  1.0|[1.06505949588608...|       1.0|
+--------------------+------------+----+-----+--------------------+----------+
only showing top 20 rows

Test Error = 0.0376884 
```
With master = local[4]. The transfer learning can finish in 8 minutes. As we can see,
the model from transfer learning can achieve a 96.3% accuracy on the validation set.
