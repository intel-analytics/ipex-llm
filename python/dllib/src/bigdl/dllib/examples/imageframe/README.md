## Overview
ImageFrame provides rich deep learning APIs for scalable image processing. This example illustrates how to do model validation with Python APIs using inception-v1 model

## Run Model validation

### Preparation

In order to run the validation application, you should prepare the validation dataset and inception-v1 model

1) Prepare dataset

This excample load ImageNet validation dataset directly from hadoop sequence file, for how to prepare the sequence file, please check [here](../../../models/inception#prepare-the-data)

2) Download pre-trained inception model

BigDL provides a rich set of pre-trained models, please check [BigDL Models](https://github.com/intel-analytics/analytics-zoo/tree/master/models) for details

Download inception-v1 model by running

wget https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_inception-v1_imagenet_0.4.0.model

### Run validation program

Run the program as a spark application with below command in standalone mode

```shell
master=spark://xxx.xxx.xxx.xxx:xxxx # please set your own spark master
imageFolder=hdfs://...
pathToModel=... #set path to your downloaded model
batchsize=... #total batch size
PYTHON_API_ZIP_PATH=dist/lib/bigdl-VERSION-python-api.zip
BigDL_JAR_PATH=dist/lib/bigdl-VERSION-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
${SPARK_HOME}/bin/spark-submit \
        --master $master \
        --driver-cores 10  \
        --driver-memory 5g  \
        --total-executor-cores 20  \
        --executor-cores 20  \
        --executor-memory 20g \
        --py-files ${PYTHON_API_ZIP_PATH},inception_validation.py  \
        --properties-file dist/conf/spark-bigdl.conf \
        --jars ${BigDL_JAR_PATH} \
        --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
        --conf spark.executor.extraClassPath=dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
        inception_validation.py ${imageFolder} ${pathToModel} ${batchsize}
```
