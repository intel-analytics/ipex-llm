# Image Classification example
This example illustrates how to do the image classification with pre-trained model

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via pip or download the prebuilt package.

## Prepare pre-trained models
Download pre-trained models from [Image Classification](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/image-classification.md)

## Prepare predict dataset
Put your image data for prediction in one folder.

## Run after pip install
You can easily use the following commands to run this example:
```bash
export SPARK_DRIVER_MEMORY=10g
modelPath=... // model path. Local file system/HDFS/Amazon S3 are supported
imagePath=... // image path. Local file system/HDFS are supported. With local file system, the files need to be available on all nodes in the cluster and please use file:///... for local files.
topN=... // top n prediction
partitionNum=... // A suggestion value of the minimal partition number
python path/to/predict.py -f $imagePath --model $modelPath --topN 5 --partition_num ${partitionNum}
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.

## Run with prebuilt package
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
modelPath=... // model path. Local file system/HDFS/Amazon S3 are supported
imagePath=... // image path. Local file system/HDFS are supported. With local file system, the files need to be available on all nodes in the cluster and please use file:///... for local files.
topN=... // top n prediction
partitionNum=... // A suggestion value of the minimal partition number

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master local[4] \
    --driver-memory 10g \
    --executor-memory 10g \
    path/to/predict.py -f $imagePath --model $modelPath --topN 5 --partition_num ${partitionNum}
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.
