# AlexNet Model on Imagenet
This example show how to use a spark cluster to train a AlexNet 
model. The model is proposed in [this paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

## Get the BigDL Files
You can build BigDL refer to [Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page) from the source code
 
## Prepare the data
You can download one from [here]() or build one by refer to the [Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page) from the source code.

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
cat bigdl_folder/classes.lst | while read CLASS_NAME; do mkdir -p ${CLASS_NAME}; done
cat bigdl_folder/img_class.lst | while read PARAM; do mv ${PARAM/ n[0-9]*/} ${PARAM/ILSVRC*JPEG /}/; done
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
Example command:
```bash
./dist/bin/bigdl.sh -- java -cp bigdl_folder/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.alexnet.Train --env local --batchSize batch_size --core core_number --nodeNumber 1 -f sequence_files_folder --checkpoint .
```

On Spark
```bash
./dist/bin/bigdl.sh -- spark-submit --driver-class-path bigdl_folder/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --class com.intel.analytics.bigdl.models.alexnet.Train dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --env spark --batchSize batch_size --core core_number --nodeNumber node_number -f hdfs://imagenet_path/imagenet --checkpoint .
```

## Test the Model
Example command
```bash
./dist/bin/bigdl.sh -- java -cp ./dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-all-in-one.jar com.intel.analytics.bigdl.models.alexnet.Test -f path_val_data/ --model model.file -c core_number -n 1 -b batch_size --env local
```

On Spark
```bash
./dist/bin/bigdl.sh -- spark-submit --driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --class com.intel.analytics.bigdl.models.alexnet.Test dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --env spark --batchSize batch_size --core core_number --nodeNumber node_number -f hdfs://imagenet_path/imagenet/val --model model.file
```