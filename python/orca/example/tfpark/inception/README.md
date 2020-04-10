# Inception Model on Imagenet
This example demonstrates how to use Analytics-zoo to train a TensorFlow [Inception v1](https://arxiv.org/abs/1409.4842) model on the [ImageNet](http://image-net.org/index) data.
## Get the JAR
You can build one by refer to the
[Build Page](https://analytics-zoo.github.io/master/#ScalaUserGuide/install/#download-analytics-zoo-source) from the source code. We
will release a pre-build package soon.

## Prepare the data
You can download imagenet-2012 data from <http://image-net.org/download-images> and put them in the directory containing this readme.
 
After you download the files(**ILSVRC2012_img_train.tar** and **ILSVRC2012_img_val.tar**), 
run the following commands to prepare the data.

The first arguments of `prepare_data.sh` is the output partition number of the sequence files, which is recommended to be the number of cores of the machine executing this script.

Please prepare at least 1 TB of space for this part.

```bash
bash prepare_data.sh $parition_num
```

This command will generate the hadoop sequence files in the `sequence` folder.

Then you can put the sequence files to hdfs. E.g.

```bash
hadoop fs -mkdir -p /user/root
hadoop fs -copyFromLocal sequence/ /user/root/
```

## Train the Model
* Spark standalone, example command
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project
# You can uncomment the following line if hyper-threading is disabled in your cluster
# export KMP_AFFINITY=granularity=fine,verbose,compact
DATA_PATH=hdfs://[IP:port]/path/to/sequence/files
mkdir -p /tmp/models/
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
--master spark://xxx.xxx.xxx.xxx:xxxx \  
--executor-cores 54 \  
--total-executor-cores 224 \  
--executor-memory 175G \ 
--driver-memory 20G \ 
--conf spark.network.timeout=10000000  inception.py \ 
--batchSize 1792 \
--learningRate 0.0896 \
-f $DATA_PATH \
--checkpoint /tmp/models/inception \ 
--maxIteration 62000
```

* Spark yarn client mode, example command
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project
# You can uncomment the following line if hyper-threading is disabled in your cluster
# export KMP_AFFINITY=granularity=fine,verbose,compact
DATA_PATH=hdfs://[IP:port]/path/to/sequence/files
mkdir -p /tmp/models/
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
--master yarn \
--deploy-mode client \
--executor-cores 54 \
--num-executors 4 \ 
--executor-memory 175G \ 
--driver-memory 20G \ 
--conf spark.network.timeout=10000000  inception.py \ 
--batchSize 1792 \
--learningRate 0.0896 \
-f $DATA_PATH \
--checkpoint /tmp/models/inception \
--maxIteration 62000
```

In the above commands
* -f: where you put your ImageNet data, it should be a hdfs folder
* --checkpoint: Where you cache the model/train_state snapshot. You should input a folder and
make sure the folder is created when you run this example. The model snapshot will be named as
model.#iteration_number, and train state will be named as optimMethod.#iteration_number. Note that if
there are some files already exist in the folder, the old file will not be overwrite for the
safety of your model files.
* --batchSize: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number *
core_number. In this example, node_number is 1 and the mini-batch size is suggested to be set to core_number * 4
* --learningRate: inital learning rate. Note in this example, we use a Poly learning rate decay
policy.
* --weightDecay: weight decay.
* --checkpointIteration: the checkpoint interval in iteration.
* --maxLr: optional. Max learning rate after warm up. It has to be set together with warmupEpoch.
* --warmupEpoch: optional. Epoch numbers need to take to increase learning rate from learningRate to maxLR.
* --maxIteration: max iteration
