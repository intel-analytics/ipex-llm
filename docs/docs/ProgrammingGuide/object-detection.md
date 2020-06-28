Analytics Zoo provides a collection of pre-trained models for Object Detection. These models can be used for out-of-the-box inference if you are interested in categories already in the corresponding datasets. According to the business scenarios, users can embed the models locally, distributedly in Apache Spark, Apache Storm or Apache Flink.

## Object Detection examples

Analytics Zoo provides two typical kind of pre-trained Object Detection models : [SSD](https://arxiv.org/abs/1512.02325) and [Faster-RCNN](https://arxiv.org/abs/1506.01497) on dataset [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](http://cocodataset.org/#home). For the usage of these models, please check below examples.


**Scala**


[Scala example](https://github.com/intel-analytics/analytics-zoo/blob/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/objectdetection/inference/Predict.scala)

It's very easy to apply the model for inference with below code piece.

```scala
val model = ObjectDetector.load[Float](params.model)
val data = ImageSet.read(params.image, sc, params.nPartition)
val output = model.predictImageSet(data)
```

For preprocessors for Object Detection models, please check [Object Detection Config](https://github.com/intel-analytics/analytics-zoo/blob/master/zoo/src/main/scala/com/intel/analytics/zoo/models/image/objectdetection/ObjectDetectionConfig.scala)

Note: We expect the loaded images has 3 channels. If the channel is not 3(eg, gray/png images), please set `imageCodec` when loading images `ImageSet.read`. See https://analytics-zoo.github.io/0.1.0/#ProgrammingGuide/object-detection/#object-detection-examples

Users can also do the inference directly using Analytics zoo.
Sample code for SSD VGG on PASCAL as below:

```scala
val model = ObjectDetector.load[Float](params.model)
val data = ImageSet.read(params.image, sc, params.nPartition)
val preprocessor = Resize(300, 300) ->
                         ChannelNormalize(123f, 117f, 104f, 1f, 1f, 1f) ->
                         MatToTensor() -> ImageFrameToSample()
val output = model.predictImageset(data)
```

**Python**

[Python example](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/objectdetection)

It's very easy to apply the model for inference with below code piece.
```
model = ObjectDetector.load_model(model_path)
image_set = ImageSet.read(img_path, sc)
output = model.predict_image_set(image_set)
```

User can also define his own configuration to do the inference with below code piece.
```
model = ObjectDetector.load_model(model_path)
image_set = ImageSet.read(img_path, sc)
preprocessing = ChainedPreprocessing(
                [ImageResize(256, 256), ImageCenterCrop(224, 224),
                ImageChannelNormalize(123.0, 117.0, 104.0), ImageMatToTensor(),
                ImageSetToSample()])
config = ImageConfigure(preprocessing)
output = model.predict_image_set(image_set)
```

For preprocessors for Object Detection models, please check [Object Detection Config](https://github.com/intel-analytics/zoo/blob/master/zoo/src/main/scala/com/intel/analytics/zoo/models/image/objectdetection/ObjectDetectionConfig.scala)

## Download link

**PASCAL VOC models**

* [SSD 300x300 MobileNet](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model)
* [SSD 300x300 VGG](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_PASCAL_0.1.0.model)
* [SSD 300x300 VGG Quantize](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300-quantize_PASCAL_0.1.0.model)
* [SSD 512x512 VGG](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-512x512_PASCAL_0.1.0.model)
* [SSD 512x512 VGG Quantize](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-512x512-quantize_PASCAL_0.1.0.model)
* [Faster-RCNN VGG](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_frcnn-vgg16_PASCAL_0.1.0.model)
* [Faster-RCNN VGG Compress](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_frcnn-vgg16-compress_PASCAL_0.1.0.model)
* [Faster-RCNN VGG Compress Quantize](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_frcnn-vgg16-compress-quantize_PASCAL_0.1.0.model)
* [Faster-RCNN PvaNet](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_frcnn-pvanet_PASCAL_0.1.0.model)
* [Faster-RCNN PvaNet Compress](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_frcnn-pvanet-compress_PASCAL_0.1.0.model)
* [Faster-RCNN PvaNet Compress Quantize](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_frcnn-pvanet-compress-quantize_PASCAL_0.1.0.model)


**COCO models**

* [SSD 300x300 VGG](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model)
* [SSD 300x300 VGG Quantize](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300-quantize_COCO_0.1.0.model)
* [SSD 512x512 VGG](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-512x512_COCO_0.1.0.model)
* [SSD 512x512 VGG Quantize](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-512x512-quantize_COCO_0.1.0.model)
