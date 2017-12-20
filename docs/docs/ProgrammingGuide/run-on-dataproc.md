---

---
## **Before You Start**

Before using BigDL on Dataproc, you need setup a project and create a cluster on Dataproc(you may refer to [https://cloud.google.com/sdk/docs/how-to](https://cloud.google.com/dataproc/docs/how-to) for more instructions).  

Please disable spark.dynamicAllocation when create cluster.  
eg. gcloud dataproc clusters create bigdl --project $PROJECT_NAME --worker-machine-type $MACHINETYPE --master-machine-type $MACHINETYPE --num-workers $WORKERNUMBER --properties spark:spark.dynamicAllocation.enabled=false   

 Or please set --conf "spark.dynamicAllocation.enabled=false" when submit spark jobs


---
## **Download BigDL**

BigDL can be downloads from https://bigdl-project.github.io/master/#release-download. It provides prebuild binary for different Spark version. Please ssh on the master VM Instances and download BigDL according to Spark version. 
```bash
wget BIGDL_LINK
```

After downloaded, you will be able to find a zip file, unzip the file
```bash
unzip xxx.zip
```

---
## **Run BigDL Scala Examples**


You can run BigDL examples using the `run.example.sh` script with following parameters:

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

eg: ./bin/run.example.sh --model lenet --nodes 2 --cores 2 --memory 1g --batch-size 16 -j lib/$BIGDLJAR -p spark_buildIn

You can also run lenet examples in below command:   
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
wget https://github.com/intel-analytics/BigDL/blob/master/pyspark/bigdl/models/lenet/lenet5.py

MASTER=yarn-client
PYTHON_API_ZIP_PATH=./lib/$BIGDLJAR
BigDL_JAR_PATH=./lib/$BIGDLJAR
PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 2g  \
        --num-executors 2  \
        --executor-cores 2  \
        --executor-memory 4g \
        --py-files ${PYTHON_API_ZIP_PATH},./BigDL/pyspark/bigdl/models/lenet/lenet5.py  \
        --properties-file ./conf/spark-bigdl.conf \
        --jars ${BigDL_JAR_PATH} \
        --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
        --conf spark.executor.extraClassPath=${BigDL_JAR_PATH} \
        ./BigDL/pyspark/bigdl/models/lenet/lenet5.py \
        --action train \
        --dataPath ./mnist
```