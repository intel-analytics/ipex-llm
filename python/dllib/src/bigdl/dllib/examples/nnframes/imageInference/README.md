## Summary

Python demo of image classification: inference with a pre-trained Inception_V1 model based on Spark DataFrame (Dataset).

Zoo provides the DataFrame-based API for image reading, preprocessing, model training and inference. The related classes followed the typical estimator/transformer pattern of Spark ML and can be used in a standard Spark ML pipeline.

## Install or download Analytics Zoo

Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Image Model Inference
You can run ModelInference example by the following steps.

1. Get the pre-trained Inception-V1 model
Download the pre-trained Inception-V1 model from [Analytics Zoo](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_inception-v1_imagenet_0.4.0.model),
and put it in `/tmp/zoo` or other path.

2. Prepare predict dataset
You can use your own image data (JPG or PNG), or some images from imagenet-2012 validation
dataset <http://image-net.org/download-images> to run the example. We use `/tmp/zoo/infer_images`
in this example.

3. Run this example
ImageInferenceExample.py takes 2 parameters: Path to the pre-trained models and path to the images.

- Run after pip install
You can easily use the following commands to run this example:
    ```bash
    export SPARK_DRIVER_MEMORY=3g
    python ImageInferenceExample.py /tmp/zoo/bigdl_inception-v1_imagenet_0.4.0.model /tmp/zoo/infer_images
    ```
    See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.

- Run with prebuilt package
Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode:
    ```bash
    export SPARK_HOME=the root directory of Spark
    export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

    ${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master local[1] \
    --driver-memory 3g \
    ImageInferenceExample.py \
    /tmp/zoo/bigdl_inception-v1_imagenet_0.4.0.model /tmp/zoo/infer_images
    ```
    See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.

4. see the result
After inference, you should see something like this in the console:
```
+-------------------------------------------------------+----------+
|name                                                   |prediction|
+-------------------------------------------------------+----------+
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000001.JPEG|59.0      |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000002.JPEG|796.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000003.JPEG|231.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000004.JPEG|970.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000005.JPEG|432.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000006.JPEG|59.0      |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000007.JPEG|378.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000008.JPEG|713.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000009.JPEG|107.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000010.JPEG|284.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000011.JPEG|110.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000012.JPEG|287.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000013.JPEG|371.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000014.JPEG|758.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000015.JPEG|596.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000016.JPEG|148.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000017.JPEG|2.0       |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000018.JPEG|22.0      |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000019.JPEG|479.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000020.JPEG|518.0     |
+-------------------------------------------------------+----------+
only showing top 20 rows

```
    To map the class to human readable text, please refer to https://github.com/Lasagne/Recipes/blob/master/examples/resnet50/imagenet_classes.txt 
