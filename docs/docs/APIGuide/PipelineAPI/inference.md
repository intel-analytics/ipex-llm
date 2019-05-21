## Inference Model

## Overview

Inference is a package in Analytics Zoo aiming to provide high level APIs to speed-up development. It 
allows user to conveniently use pre-trained models from Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).
Inference provides multiple Scala interfaces.



## Highlights

1. Easy-to-use APIs for loading and prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).

2. Support transformation of various input data type, thus supporting future prediction tasks.

3. Transparently support the OpenVINO toolkit, which deliver a significant boost for inference speed ([up to 19.9x](https://software.intel.com/en-us/blogs/2018/05/15/accelerate-computer-vision-from-edge-to-cloud-with-openvino-toolkit)).

## Primary APIs for Java

**load**

AbstractInferenceModel provides `load` API for loading a pre-trained model,
thus we can conveniently load various kinds of pre-trained models in java applications. The load result of
`AbstractInferenceModel` is an `AbstractModel`.
We just need to specify the model path and optionally weight path if exists where we previously saved the model.

***load***

`load` method is to load a BigDL model.

***loadCaffe***

`loadCaffe` method is to load a caffe model.

***loadTF***

`loadTF` method is to load a tensorflow model. There are two backends to load a tensorflow model and to do the predictions: TFNet and OpenVINO. For OpenVINO backend, [supported tensorflow models](https://docs.openvinotoolkit.org/2018_R5/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) are listed below:

    inception_v1
    inception_v2
    inception_v3
    inception_v4
    inception_resnet_v2
    mobilenet_v1
    nasnet_large
    nasnet_mobile
    resnet_v1_50
    resnet_v2_50
    resnet_v1_101
    resnet_v2_101
    resnet_v1_152
    resnet_v2_152
    vgg_16
    vgg_19
    faster_rcnn_inception_resnet_v2_atrous_coco
    faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco
    faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid
    faster_rcnn_inception_resnet_v2_atrous_oid
    faster_rcnn_nas_coco
    faster_rcnn_nas_lowproposals_coco
    faster_rcnn_resnet101_coco
    faster_rcnn_resnet101_kitti
    faster_rcnn_resnet101_lowproposals_coco
    mask_rcnn_inception_resnet_v2_atrous_coco
    mask_rcnn_inception_v2_coco
    mask_rcnn_resnet101_atrous_coco
    mask_rcnn_resnet50_atrous_coco
    ssd_inception_v2_coco
    ssd_mobilenet_v1_coco
    ssd_mobilenet_v2_coco
    ssdlite_mobilenet_v2_coco

***loadOpenVINO***

`loadOpenVINO` method is to load an OpenVINO Intermediate Representation(IR).

***loadOpenVINOInt8***

`loadOpenVINO` method is to load an OpenVINO Int8 Intermediate Representation(IR).

**predict**

AbstractInferenceModel provides `predict` API for prediction with loaded model.
The predict result of`AbstractInferenceModel` is a `List<List<JTensor>>` by default.

**predictInt8**

AbstractInferenceModel provides `predictInt8` API for prediction with loaded int8 model.
The predictInt8 result of`AbstractInferenceModel` is a `List<List<JTensor>>` by default.


## Examples

It's very easy to apply abstract inference model for inference with below code piece.
You will need to write a subclass that extends AbstractinferenceModel.
```java
import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

public class TextClassificationModel extends AbstractInferenceModel {
    public TextClassificationModel() {
        super();
    }
 }
TextClassificationModel model = new TextClassificationModel();
model.load(modelPath, weightPath);
List<List<JTensor>> result = model.predict(inputList);
```

## Primary APIs for Scala

**InferenceModel**

`InferenceModel` is a thead-safe wrapper of AbstractModels, which can be used to load models and do the predictions.

***doLoad***

`doLoad` method is to load a bigdl, analytics-zoo model.

***doLoadCaffe***

`doLoadCaffe` method is to load a caffe model.

***doLoadTF***

`doLoadTF` method is to load a tensorflow model. The model can be loaded as a `FloatModel` or an `OpenVINOModel`. There are two backends to load a tensorflow model: TFNet and OpenVINO. 

<span id="jump">For OpenVINO backend, [supported tensorflow models](https://docs.openvinotoolkit.org/2018_R5/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) are listed below:</span>
                                          
    inception_v1
    inception_v2
    inception_v3
    inception_v4
    inception_resnet_v2
    mobilenet_v1
    nasnet_large
    nasnet_mobile
    resnet_v1_50
    resnet_v2_50
    resnet_v1_101
    resnet_v2_101
    resnet_v1_152
    resnet_v2_152
    vgg_16
    vgg_19
    faster_rcnn_inception_resnet_v2_atrous_coco
    faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco
    faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid
    faster_rcnn_inception_resnet_v2_atrous_oid
    faster_rcnn_nas_coco
    faster_rcnn_nas_lowproposals_coco
    faster_rcnn_resnet101_coco
    faster_rcnn_resnet101_kitti
    faster_rcnn_resnet101_lowproposals_coco
    mask_rcnn_inception_resnet_v2_atrous_coco
    mask_rcnn_inception_v2_coco
    mask_rcnn_resnet101_atrous_coco
    mask_rcnn_resnet50_atrous_coco
    ssd_inception_v2_coco
    ssd_mobilenet_v1_coco
    ssd_mobilenet_v2_coco
    ssdlite_mobilenet_v2_coco

***doLoadOpenVINO***
                                          
`doLoadOpenVINO` method is to load an OpenVINO Intermediate Representation(IR).

***doLoadOpenVINOInt8***

`doLoadOpenVINOInt8` method is to load an OpenVINO Int8 Intermediate Representation(IR).

***doReload***

`doReload` method is to reload the bigdl, analytics-zoo model.

***doPredict***

`doPredict` method is to do the prediction.

***doPredictInt8***

`doPredict` method is to do the prediction with Int8 model. If model doesn't support predictInt8, will throw RuntimeException with `does not support predictInt8` message.

**InferenceSupportive**

`InferenceSupportive` is a trait containing several methods for type transformation, which transfer a model input 
to a valid data type, thus supporting future inference model prediction tasks.

For example, method `transferTensorToJTensor` convert a model input of data type `Tensor` 
to [`JTensor`](https://github.com/intel-analytics/analytics-zoo/blob/88afc2d921bb50341d8d7e02d380fa28f49d246b/zoo/src/main/java/com/intel/analytics/zoo/pipeline/inference/JTensor.java)
, which will be the input for a FloatInferenceModel.

**AbstractModel**

`AbstractModel` is an abstract class to provide APIs for basic functions - `predict` interface for prediction, `copy` interface for coping the model into the queue of AbstractModels, `release` interface for releasing the model and `isReleased` interface for checking the state of model release.  

**FloatModel**

`FloatModel` is an extending class of `AbstractModel` and achieves all `AbstractModel` interfaces. 

**OpenVINOModel**

`OpenVINOModel` is an extending class of `AbstractModel`. It achieves all `AbstractModel` functions.

**InferenceModelFactory**

`InferenceModelFactory` is an object with APIs for loading pre-trained Analytics Zoo models, Caffe models, Tensorflow models and OpenVINO Intermediate Representations(IR).
Analytics Zoo models, Caffe models, Tensorflow models can be loaded as FloatModels. The load result of it is a `FloatModel`
Tensorflow models and OpenVINO Intermediate Representations(IR) can be loaded as OpenVINOModels. The load result of it is an `OpenVINOModel`. 
The load result of it is a `FloatModel` or an `OpenVINOModel`. 


**OpenVinoInferenceSupportive**

`OpenVinoInferenceSupportive` is an extending object of `InferenceSupportive` and focus on the implementation of loading pre-trained models, including tensorflow models and OpenVINO Intermediate Representations(IR). 
There are two backends to load a tensorflow model: TFNet and OpenVINO. For OpenVINO backend, [supported tensorflow models](#jump) are listed in the section of `doLoadTF` method of `InferenceModel` API above. 
