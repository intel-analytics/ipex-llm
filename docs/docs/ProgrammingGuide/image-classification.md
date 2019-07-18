Analytics Zoo provides a collection of pre-trained models for Image Classification. These models can be used for out-of-the-box inference if you are interested in categories already in the corresponding datasets. According to the business scenarios, users can embed the models locally, distributedly in Spark such as Apache Storm and Apache Flink.

## Image Classification examples

Analytics Zoo provides several typical kind of pre-trained Image Classfication models : [Alexnet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networksese), [Inception-V1](https://arxiv.org/abs/1409.4842), [VGG](https://arxiv.org/abs/1409.1556), [Resnet](https://arxiv.org/abs/1512.03385), [Densenet](https://arxiv.org/abs/1608.06993), [Mobilenet](https://arxiv.org/abs/1704.04861), [Squeezenet](https://arxiv.org/abs/1602.07360) models. To use these models, please check below examples.


**Scala**


[Scala example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/imageclassification)

It's very easy to apply the model for inference with below code piece.

```scala
val imc = ImageClassifier.loadModel[Float](params.model)
val data = ImageSet.read(params.image, sc, params.nPartition)
val output = imc.predictImageSet(data)
```

User can also define his own configuration to do the inference with below code piece.

```scala
val imc = ImageClassifier.loadModel[Float](params.model)
val data = ImageSet.read(params.image, sc, params.nPartition)
val preprocessing = ImageResize(256, 256)-> ImageCenterCrop(224, 224) ->
        ImageChannelNormalize(123, 117, 104) ->
        ImageMatToTensor[Float]() ->
        ImageSetToSample[Float]()
val config = ImageConfigure[Float](preprocessing)        
val output = imc.predictImageSet(data, config)
```

**Python**

[Python example](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/imageclassification)

It's very easy to apply the model for inference with below code piece.
```
imc = ImageClassifier.load_model(model_path)
image_set = ImageSet.read(img_path, sc)
output = imc.predict_image_set(image_set)
```

User can also define his own configuration to do the inference with below code piece.
```
imc = ImageClassifier.load_model(model_path)
image_set = ImageSet.read(img_path, sc)
preprocessing = ChainedPreprocessing(
                [ImageResize(256, 256), ImageCenterCrop(224, 224),
                ImageChannelNormalize(123.0, 117.0, 104.0), ImageMatToTensor(),
                ImageSetToSample()])
config = ImageConfigure(preprocessing) 
output = imc.predict_image_set(image_set)
```

For preprocessors for Image Classification models, please check [Image Classification Config](https://github.com/intel-analytics/zoo/blob/master/zoo/src/main/scala/com/intel/analytics/zoo/models/image/imageclassification/ImageClassificationConfig.scala)

## Download link

* [Alexnet](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_alexnet_imagenet_0.1.0.model)
* [Alexnet Quantize](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_alexnet-quantize_imagenet_0.1.0.model)
* [Inception-V1](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_inception-v1_imagenet_0.1.0.model)
* [Inception-V1 Quantize](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_inception-v1-quantize_imagenet_0.1.0.model)
* [Inception-V3](https://analytics-zoo-models.s3-ap-southeast-1.amazonaws.com/analytics-zoo_inception-v3_imagenet_0.1.0.model)
* [Inception-V3 Quantize](https://analytics-zoo-models.s3-ap-southeast-1.amazonaws.com/analytics-zoo_inception-v3-quantize_imagenet_0.1.0.model)
* [VGG-16](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_vgg-16_imagenet_0.1.0.model)
* [VGG-16 Quantize](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_vgg-16-quantize_imagenet_0.1.0.model)
* [VGG-19](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_vgg-19_imagenet_0.1.0.model)
* [VGG-19 Quantize](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_vgg-19-quantize_imagenet_0.1.0.model)
* [Resnet-50](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_resnet-50_imagenet_0.1.0.model)
* [Resnet-50 Quantize](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_resnet-50-quantize_imagenet_0.1.0.model)
* [Resnet-50 Int8](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_resnet-50-int8_imagenet_0.5.0.model)
* [Densenet-161](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_densenet-161_imagenet_0.1.0.model)
* [Densenet-161 Quantize](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_densenet-161-quantize_imagenet_0.1.0.model)
* [Mobilenet](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_mobilenet_imagenet_0.1.0.model)
* [Mobilenet-V2](https://analytics-zoo-models.s3-ap-southeast-1.amazonaws.com/analytics-zoo_mobilenet-v2_imagenet_0.1.0.model)
* [Mobilenet-V2 Quantize](https://analytics-zoo-models.s3-ap-southeast-1.amazonaws.com/analytics-zoo_mobilenet-v2-quantize_imagenet_0.1.0.model)
* [Squeezenet](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_squeezenet_imagenet_0.1.0.model)
* [Squeezenet Quantize](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_squeezenet-quantize_imagenet_0.1.0.model)
