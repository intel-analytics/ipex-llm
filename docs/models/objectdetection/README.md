# Analytics Zoo Object Detection API

Analytics Zoo provides a collection of pre-trained models for Object Detection. These models can be used for out-of-the-box inference if you are interested in categories already in the corresponding datasets. According to the business scenarios, users can embed the models locally, distributedly in Spark such as Apache Storm and Apache Flink.

***Object Detection models***

Analytics Zoo provides two typical kind of pre-trained Object Detection models : [SSD](https://arxiv.org/abs/1512.02325) and [Faster-RCNN](https://arxiv.org/abs/1506.01497) on dataset [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](http://cocodataset.org/#home). For the usage of these models, please check below examples.

[Scala example](../../zoo/src/main/scala/com/intel/analytics/zoo/examples/objectdetection/Predict.scala)

It's very easy to apply the model for inference with below code piece.

```scala
val model = ObjectDetector.load[Float](params.model)
val data = ImageSet.read(params.image, sc, params.nPartition)
val output = model.predictImageSet(data)
```

For preprocessors for Object Detection models, please check [Object Detection Config](../../zoo/src/main/scala/com/intel/analytics/zoo/models/image/objectdetection/ObjectDetectionConfig.scala)

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
## Download link
### Object Detection

1. PASCAL VOC models
* [SSD 300x300 MobileNet](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model)
* [SSD 300x300 VGG](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_PASCAL_0.1.0.model)
* [SSD 300x300 VGG Quantize](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300-quantize_PASCAL_0.1.0.model)
* [SSD 512x512 VGG](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-512x512_PASCAL_0.1.0.model)
* [SSD 512x512 VGG Quantize](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-512x512-quantize_PASCAL_0.1.0.model)
* [Faster-RCNN VGG](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_frcnn-vgg16_PASCAL_0.1.0.model)
* [Faster-RCNN VGG Compress](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_frcnn-vgg16-compress_PASCAL_0.1.0.model)
* [Faster-RCNN VGG Compress Quantize](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_frcnn-vgg16-compress-quantize_PASCAL_0.1.0.model)
* [Faster-RCNN PvaNet](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_frcnn-pvanet_PASCAL_0.1.0.model)
* [Faster-RCNN PvaNet Compress](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_frcnn-pvanet-compress_PASCAL_0.1.0.model)
* [Faster-RCNN PvaNet Compress Quantize](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_frcnn-pvanet-compress-quantize_PASCAL_0.1.0.model)

2. COCO models

* [SSD 300x300 VGG](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model)
* [SSD 300x300 VGG Quantize](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300-quantize_COCO_0.1.0.model)
* [SSD 512x512 VGG](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-512x512_COCO_0.1.0.model)
* [SSD 512x512 VGG Quantize](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-512x512-quantize_COCO_0.1.0.model)
