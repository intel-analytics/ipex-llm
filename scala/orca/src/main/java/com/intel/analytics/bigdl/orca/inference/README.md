# Abstract Inference Model

## Overview

Abstract inference model is an abstract class in Analytics Zoo aiming to provide support for java implementation in loading a collection of pre-trained models(including Caffe models, Tensorflow models, OpenVINO Intermediate Representations(IR), etc.) and for model prediction. AbstractInferenceModel contains a mix of methods declared with implementation for loading models and prediction.

You will need to create a subclass which extends the AbstractInferenceModel to 
develop your java applications.

### Highlights

1. Easy-to-use java API for loading and prediction with deep learning models.
2. In a few lines, run large scale inference from pre-trained models of Analytics-Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).
3. Transparently support the OpenVINO toolkit, which deliver a significant boost for inference speed  ([up to 19.9x](https://software.intel.com/en-us/blogs/2018/05/15/accelerate-computer-vision-from-edge-to-cloud-with-openvino-toolkit)).


## Primary APIs

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

## Examples

It's very easy to apply abstract inference model for inference with below code piece. You will need to write a subclass that extends AbstractinferenceModel.
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
