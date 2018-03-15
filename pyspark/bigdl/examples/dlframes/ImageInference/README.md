# Summary

Python demo of image classification: inference with a pre-trained Inception_V1 model based on Spark DataFrame (Dataset).

BigDL provides the DataFrame-based API for image reading, preprocessing, model training and inference. The related
classes followed the typical estimator/transformer pattern of Spark ML and can be used in a standard Spark ML pipeline.

# Preparation

## Make sure Spark, BigDL are successfully installed

Please refer to [BigDL](https://bigdl-project.github.io/) for installation

## Get the imagenet ILSVRC2012 validation datasets

Download the validation dataset from http://image-net.org/download-images. For this demo, you may take the top 100 images to save time.

## Get the pre-trained Inception-V1 model

Download the pre-trained Inception-V1 model from [BigDL Zoo](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_inception-v1_imagenet_0.4.0.model)

Alternatively, user may also download pre-trained caffe/Tensorflow/keras model. Please refer to
programming guide in [BigDL](https://bigdl-project.github.io/) 

# Inference for image classification

ImageInferenceExample.py takes 2 parameters:
1. Path to the pre-trained models. (E.g. path/to/model/bigdl_inception-v1_imagenet_0.4.0.model)
2. Path to the folder of the images. (E.g. path/to/data/imagenet/validation)

User may submit ImageInferenceExample.py via spark-submit.
E.g.
```
BigDL/dist/bin/spark-submit-with-bigdl.sh --master local[1] ../bigdl/examples/dlframes/ImageInference/ImageInferenceExample.py path/to/model/bigdl_inception-v1_imagenet_0.4.0.model path/to/data/imagenet/validation
```

or run the script in Jupyter notebook or Pyspark and manually set parameters

After inference, you should see something like this in the console:

```
+-------------------------------------------------------------------------------+----------+
|name                                                                           |prediction|
+-------------------------------------------------------------------------------+----------+
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000001.JPEG|59.0      |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000002.JPEG|796.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000003.JPEG|231.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000004.JPEG|969.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000005.JPEG|521.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000006.JPEG|59.0      |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000007.JPEG|283.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000008.JPEG|697.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000009.JPEG|48.0      |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000010.JPEG|812.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000011.JPEG|956.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000012.JPEG|287.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000013.JPEG|371.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000014.JPEG|758.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000015.JPEG|596.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000016.JPEG|148.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000017.JPEG|981.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000018.JPEG|22.0      |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000019.JPEG|479.0     |
|file:/home/yuhao/workspace/data/imagenet/predict_s/ILSVRC2012_val_00000020.JPEG|518.0     |
+-------------------------------------------------------------------------------+----------+

```
To map the class to human readable text, please refer to https://github.com/Lasagne/Recipes/blob/master/examples/resnet50/imagenet_classes.txt 
