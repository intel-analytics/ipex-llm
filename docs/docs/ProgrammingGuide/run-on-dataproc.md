
## **Deploy BigDL on Dataproc**

Before using BigDL on Google Dataproc, you need setup a project and create a cluster on Dataproc(you may refer to [https://cloud.google.com/sdk/docs/how-to](https://cloud.google.com/dataproc/docs/how-to) for more instructions). Now you can create a Cloud Dataproc cluster using the Google Cloud SDK's(https://cloud.google.com/sdk/docs/) gcloud command-line tool.

To make it easy to try out BigDL on Spark on Dataproc, an initial action script is provided. You can use use this initialization action to create a new Dataproc cluster with BigDL pre-installed by https://github.com/GoogleCloudPlatform/dataproc-initialization-actions/blob/master/bigdl/bigdl.sh
You can use below command to create a new cluster with BigDL preinstall
```bash
gcloud dataproc clusters create <CLUSTER_NAME> --initialization-actions=gs://dataproc-initialization-actions/bigdl/bigdl.sh
```

Note:

By default, it will automatically download BigDL 0.4.0 for Dataproc 1.2 (Spark 2.2.0 and Scala 2.11.8). To download a different version of BigDL or one targeted to a different version of Spark/Scala, find the download URL from the BigDL releases page, and set the metadata key "bigdl-download-url".
eg.
```bash
gcloud dataproc clusters create <CLUSTER_NAME> \
    --image-version 1.0 \
    --initialization-actions gs://dataproc-initialization-actions/bigdl/bigdl.sh \
    --initialization-action-timeout 10m \
    --metadata 'bigdl-download-url=https://s3-ap-southeast-1.amazonaws.com/bigdl-download/dist-spark-1.6.2-scala-2.10.5-all-0.4.0-dist.zip'
```
More information please refer https://github.com/GoogleCloudPlatform/dataproc-initialization-actions/tree/master/bigdl

Once the cluster is provisioned, you will be able to see the cluster running in the Google Cloud Platform Console. Now you can SSH to the master node.

Cloud Dataproc support various way to SSH to the master, here we use SSH from Google Cloud SDK.
E.g.,
```bash
gcloud compute --project <PROJECT_ID> ssh --zone <ZONE> <CLUSTER_NAME>
```
Google cloud SDK will perform the authentication for you and open an SSH client (Eg Putty).

You should be able to find BigDL is located under /opt/intel-bigdl. And your VM is ready for running deep learning examples at scale!

---
## **Run BigDL Scala Examples**


Now you can run BigDL examples on Google Dataproc. For instance, you may use the `run.example.sh` script which is located under ./bin directory with following parameters:

* Mandatory parameters:
  
    * `-m|--model` which model to train, including
    
        * lenet: train the [LeNet](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/lenet) example
    
        * vgg: train the [VGG](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/vgg) example

        * inception-v1: train the [Inception v1](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/inception) example

        * perf: test the training speed using the [Inception v1](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/inception/Inception_v1.scala) model with dummy data

    * `-s|--spark-url` the master URL for the Spark cluster

    * `-n|--nodes` number of Spark slave nodes

    * `-o|--cores` number of cores used on each node

    * `-r|--memory` memory used on each node, e.g. 200g

    * `-b|--batch-size` batch size when training the model; it is expected to be a multiple of "nodes * cores"

    * `-f|--hdfs-data-dir` HDFS directory for the input images (for the "inception-v1" model training only)

* Optional parameters:

    * `-e|--max-epoch` the maximum number of epochs (i.e., going through all the input data once) used in the training; default to 90 if not specified

    * `-p|--spark` by default the example will run with Spark 1.5 or 1.6; to use Spark 2.0, please specify "spark_2.0" here (it is highly recommended to use _**Java 8**_ when running BigDL for Spark 2.0, otherwise you may observe very poor performance)

    * `-l|--learning-rate` by default the the example will use an initial learning rate of "0.01"; you can specify a different value here

After the training, you can check the log files and generated models  

Replace $BIGDLJAR with bigdl binary name in ./lib in below command, eg: bigdl-SPARK_2.2-0.3.0-jar-with-dependencies.jar  

```bash
./bin/run.example.sh --model lenet --nodes 2 --cores 2 --memory 1g --batch-size 16 -j lib/$BIGDLJAR -p spark_buildIn
```

You can also run lenet examples in below command. Before submit below command, please make sure you have already downloaded mnist and put it under mnist directory, more detail see https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/lenet:   
```bash
spark-submit \
 --executor-cores 2 \
 --num-executors 2 \
 --driver-class-path ./lib/$BIGDLJAR \
 --class com.intel.analytics.bigdl.models.lenet.Train \
 ./lib/$BIGDLJAR \
 -f ./mnist \
 -b 16
```
---
## **Run BigDL Python example**
Download lenet5.py from https://github.com/intel-analytics/BigDL/blob/master/pyspark/bigdl/models/lenet/lenet5.py
```bash 
wget https://raw.githubusercontent.com/intel-analytics/BigDL/master/pyspark/bigdl/models/lenet/lenet5.py
```

Replace $BIGDLJAR with bigdl binary name in ./lib, eg: bigdl-SPARK_2.2-0.3.0-jar-with-dependencies.jar  
Replace $BIGDL_PYTHON_ZIP with bigdl python binary name in ./lib, eg: bigdl-0.3.0-python-api.zip
```bash
PYTHON_API_ZIP_PATH=./lib/$BIGDL_PYTHON_ZIP
BigDL_JAR_PATH=./lib/$BIGDLJAR
PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
spark-submit \
        --driver-cores 2  \
        --driver-memory 2g  \
        --num-executors 2  \
        --executor-cores 2  \
        --executor-memory 4g \
        --py-files ${PYTHON_API_ZIP_PATH},./lenet5.py  \
        --properties-file ./conf/spark-bigdl.conf \
        --jars ${BigDL_JAR_PATH} \
        --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
        --conf spark.executor.extraClassPath=${BigDL_JAR_PATH} \
        ./lenet5.py \
        --action train
```