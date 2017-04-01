# Inception Model on Imagenet
This example demonstrates how to use BigDL to train and evaluate [Inception v1](https://arxiv.org/abs/1409.4842) architecture on the [ImageNet](http://image-net.org/index) data.
## Get the JAR
You can build one by refer to the
[Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page) from the source code. We
will release a pre-build package soon.

## Prepare the data
You can download imagenet-2012 data from <http://image-net.org/download-images>.
 
After you download the files(**ILSVRC2012_img_train.tar** and **ILSVRC2012_img_val.tar**), 
run the follow commands to prepare the data.

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

Now all the images belong to the same category are moved to the same folder.

This command will transform the images into hadoop sequence files, which are 
more suitable for a distributed training.

```bash
java -cp bigdl_folder/lib/bigdl-VERSION-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator -f imagenet_folder -o output_folder -p cores_number
```

It will generate the hadoop sequence files in the output folder.

## Train the Model
* Spark standalone, example command
```bash
./dist/bin/bigdl.sh -- \
spark-submit \
--master spark://xxx.xxx.xxx.xxx:xxxx \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.inception.TrainInceptionV1 \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--batchSize batch_size \
--learningRate learningRate \
-f hdfs://.../imagenet \
--checkpoint ~/models
```
* Spark yarn client mode, example command
```bash
./dist/bin/bigdl.sh -- \
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores cores_per_executor \
--num-executors executors_number \
--driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.inception.TrainInceptionV1 \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--batchSize batch_size \
--learningRate learningRate \
--weightDecay weightDecay \
--checkpointIteration checkpointIteration \
-f hdfs://.../imagenet \
--checkpoint ~/models
```
In the above commands
* -f: where you put your ImageNet data, it should be a hdfs folder
* --checkpoint: Where you cache the model/train_state snapshot. You should input a folder and
make sure the folder is created when you run this example. The model snapshot will be named as
model.#iteration_number, and train state will be named as state.#iteration_number. Note that if
there are some files already exist in the folder, the old file will not be overwrite for the
safety of your model files.
* --batchSize: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number *
core_number. In this example, node_number is 1 and the mini-batch size is suggested to be set to core_number * 4
* --learningRate: inital learning rate. Note in this example, we use a Poly learning rate decay
policy.
* --weightDecay: weight decay.
* --checkpointIteration: the checkpoint interval in iteration.

## Test the Model
* Spark standalone, example command
```bash
./dist/bin/bigdl.sh -- \
spark-submit \
--master spark://xxx.xxx.xxx.xxx:xxxx \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.inception.Test \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--batchSize batch_size \
-f hdfs://.../imagenet/val \
--model model.file
```
* Spark yarn client mode, example command
```bash
./dist/bin/bigdl.sh -- \
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores cores_per_executor \
--num-executors executors_number \
--driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.inception.Test \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--batchSize batch_size \
-f hdfs://.../imagenet/val \
--model model.file
```
In the above command
* -f: where you put your ImageNet data, it should be a hdfs folder
* --model: the model snapshot file
* --batchSize: The mini-batch size. It is expected that the mini-batch size is a multiple of
node_number * core_number. In this example, node_number is 1 and the mini-batch size is suggested to be set to core_number * 4
