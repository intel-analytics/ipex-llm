# Inception Model on ImageNet
This example demonstrates how to use BigDL to train and evaluate [Inception v1](https://arxiv.org/abs/1409.4842) (or [Inception v2](https://arxiv.org/abs/1502.03167)) architecture on the [ImageNet](http://image-net.org/index) data.

## Get the BigDL Files
You can build BigDL refer to [Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page) from the source code  

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
java -cp bigdl_folder/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator -f imagenet_folder -o output_folder -p cores_number
```

It will generate the hadoop sequence files in the output folder.

## Train the Model
Here is an example command run on Spark cluster.
```bash
./bigdl_folder/bin/bigdl.sh -- spark-submit --driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --class com.intel.analytics.bigdl.models.inception.TrainInceptionV1 dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --env spark --batchSize batch_size --core core_number --nodeNumber node_number --learningRate 0.0898 -f hdfs://.../imagenet --checkpoint .
```

## Test the Model
Example command
```bash
./bigdl_folder/bin/bigdl.sh -- spark-submit --driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --class com.intel.analytics.bigdl.models.inception.Test dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --env spark --batchSize batch_size --core core_number --nodeNumber node_number -f hdfs://.../imagenet1/val --model model.file
```
