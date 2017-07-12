## Summary
This example demonstrates how to use BigDL to load a BigDL or [Torch](http://torch.ch/) model trained on [ImageNet](http://image-net.org/) data, and then apply the loaded model to classify the contents of a set of images in Spark ML pipeline.

## Preparation

To start with this example, you need prepare your model and dataset.

1. Prepare model.

    The Torch ResNet model used in this example can be found in [Resnet Torch Model](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained).
    The BigDL Inception model used in this example can be trained with [BigDL Inception](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/inception).
    You can choose one of them, and then put the trained model in $modelPath, and set corresponding $modelType（torch or bigdl）.

2. Prepare predict dataset

    Put your image data for prediction in the ./predict folder. Alternatively, you may also use imagenet-2012 validation dataset to run the example, which can be found from <http://image-net.org/download-images>. After you download the file (ILSVRC2012_img_val.tar), run the follow commands to prepare the data.

    ```bash
    mkdir predict
    tar -xvf ILSVRC2012_img_val.tar -C ./predict/
    ```


     <code>Note: </code>For large dataset, you may want to read image data from HDFS.This command will transform the images into hadoop sequence files:

```bash
mkdir -p val/images
mv predict/* val/images/
java -cp bigdl_folder/lib/bigdl-VERSION-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator -f ./ --validationOnly --hasName
mv val/*.seq predict/
```


## Run this example

Command to run the example in Spark standalone mode:
```
        BigDL_HOME=...
        SPARK_HOME=...
        MASTER=...
        PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
        BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
        PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
        source ${BigDL_HOME}/dist/bin/bigdl.sh

        ${SPARK_HOME}/bin/spark-submit \
            --master ${MASTER} \
            --driver-cores 4  \
            --driver-memory 10g  \
            --total-executor-cores 4  \
            --executor-cores 4  \
            --executor-memory 20g \
            --conf spark.akka.frameSize=64 \
            --py-files ${PYTHON_API_ZIP_PATH},${BigDL_HOME}/pyspark/dl/example/image_classification/image_classification.py  \
            --jars ${BigDL_JAR_PATH} \
            --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
            --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
            ${BigDL_HOME}/pyspark/dl/example/image_classification/image_classification.py \
             --modelPath ./resnet-18.t7 \
             --folder ./predict \
             --modelType torch \
             --isHdfs false
```

where

* ```--modelPath``` is the path to the model file.
* ```--folder``` is the folder of predict images.
* ```--modelType``` is the type of model to load, it can be ```bigdl``` or ```torch```.
* ```--showNum``` is the result number to show, default 100.
* ```--isHdfs``` is the type of predict data. "true" means reading sequence file from hdfs, "false" means reading local images, default "false".
