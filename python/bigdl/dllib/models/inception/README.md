# Inception Model on Imagenet
This example demonstrates how to use BigDL to train [Inception v1](https://arxiv.org/abs/1409.4842) architecture on the [ImageNet](http://image-net.org/index) data.
## Get the JAR
You can build one by refer to the
[Build Page](https://bigdl-project.github.io/master/#ScalaUserGuide/install-build-src/) from the source code. We
will release a pre-build package soon.

## Prepare the data
You can download imagenet-2012 data from <http://image-net.org/download-images>.
 
After you download the files(**ILSVRC2012_img_train.tar** and **ILSVRC2012_img_val.tar**), 
run the following commands to prepare the data.

```bash
mkdir train
mv ILSVRC2012_img_train.tar train/
cd train
tar -xvf ILSVRC2012_img_train.tar
rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read CLASS_NAME ; do mkdir -p "${CLASS_NAME%.tar}"; tar -xvf "${CLASS_NAME}" -C "${CLASS_NAME%.tar}"; done
rm *.tar
cd ../
mkdir val
mv ILSVRC2012_img_val.tar val/
cd val
tar -xvf ILSVRC2012_img_val.tar
cat classes.lst | while read CLASS_NAME; do mkdir -p ${CLASS_NAME}; done
cat img_class.lst | while read PARAM; do mv ${PARAM/ n[0-9]*/} ${PARAM/ILSVRC*JPEG /}; done
rm ILSVRC2012_img_val.tar
```

Now all the images belonging to the same category are moved to the same folder.

This command will transform the images into hadoop sequence files, which are 
more suitable for a distributed training.

```bash
java -cp bigdl_source_folder/spark/dl/target/bigdl-VERSION-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator -f imagenet_folder -o output_folder -p cores_number
```

It will generate the hadoop sequence files in the output folder.

## Train the Model
* Spark standalone, example command
```
BigDL_HOME=...
SPARK_HOME=...
PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_PATH}:$PYTHONPATH

${SPARK_HOME}/bin/spark-submit \
--master spark://... \
--executor-memory 150g \
--driver-memory 100g \
--executor-cores 4 \
--total-executor-cores 64  \
--properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
--jars ${BigDL_JAR_PATH} \
--py-files ${PYTHON_API_PATH} \
${BigDL_HOME}/pyspark/bigdl/models/inception/inception.py \
-f hdfs://... \
--batchSize 1024 \
--learningRate 0.065 \
--weightDecay 0.0002 \
--checkpointIteration 1000 \
-i 90000 \
--checkpoint /models/inception
```
* Spark yarn client mode, example command
```
BigDL_HOME=...
SPARK_HOME=...
PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_PATH}:$PYTHONPATH

${SPARK_HOME}/bin/spark-submit \
--master yarn \
--deploy-mode client \
--executor-memory 150g \
--driver-memory 100g \
--executor-cores 4 \
--num-executors 16 \
--properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
--jars ${BigDL_JAR_PATH} \
--conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
--conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
--py-files ${PYTHON_API_PATH} \
${BigDL_HOME}/pyspark/bigdl/models/inception/inception.py \
-f hdfs://... \
--batchSize 1024 \
--learningRate 0.065 \
--weightDecay 0.0002 \
--checkpointIteration 1000 \
-i 90000 \
--checkpoint /models/inception

```

In the above commands
* -f: where you put your ImageNet data, it should be a hdfs folder
* --checkpoint: Where you cache the model/train_state snapshot. You should input a folder and
make sure the folder is created when you run this example. The model snapshot will be named as
model.#iteration_number, and train state will be named as optimMethod.#iteration_number. Note that if
there are some files already exist in the folder, the old file will not be overwritten for the
safety of your model files.
* --batchSize: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number *
core_number. In this example, node_number is 1 and the mini-batch size is suggested to be set to core_number * 4
* --learningRate: inital learning rate. Note in this example, we use a Poly learning rate decay
policy.
* --weightDecay: weight decay.
* -i: max iteration
* --checkpointIteration: the checkpoint interval in iteration.