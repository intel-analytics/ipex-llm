# MaskRCNN
This example demonstrates how to use BigDL to evaluate the [MaskRCNN](https://arxiv.org/abs/1703.06870) architecture on COCO data

## Prepare the data
* You can download [COCO dataset](<http://cocodataset.org/>) firstly.
Extract the dataset and get images and annotations like (use **coco_2017_val** as example):
```
coco
|_ coco_val2017
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ annotations
   |_ instances_train2017.json
   |_ ...
```

* Generate the hadoop sequence files for COCO dataset
The following command will transform the images and annotations into hadoop sequence files.
```bash
java -cp com.intel.analytics.bigdl.models.utils.COCOSeqFileGenerator bigdl-VERSION-jar-with-dependencies.jar -f ./coco/coco_val2017 -m ./coco/annotations/instances_val2017.json -p 4 -o ./coco/output
```
In the above commands:
-f: the COCO image files location
-m: the annotation json file location
-o: generated seq files location
-p: number of parallel

## Data Processing
Input data are transformed by several pipeline classes, such as ScaleResize, ChannelNormalize, ImageFeatureToBatch, etc.

## Model
You can download **preTrain-MaskRCNN model** for BigDL by running
```bash
wget https://bigdlmodels.s3-us-west-2.amazonaws.com/segmentation/bigdl_mask-rcnn_COCO_0.10.0.model
```
This MaskRCNN model refers to [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), and the model backbone is **R-50-FPN**.

## Test the Model
* Spark standalone, example command
```bash
spark-submit \
--master spark://xxx.xxx.xxx.xxx:xxxx \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.maskrcnn.Test \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--batchSize batch_size \
-f hdfs://.../coco/val \
--model modelPath
```
* Spark yarn client mode, example command
```bash
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores cores_per_executor \
--num-executors executors_number \
--driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.inception.Test \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--batchSize batch_size \
-f hdfs://.../coco/val \
--model modelPath
```
In the above command
* -f: where you put your COCO data, it should be a hdfs folder
* --model: the model snapshot file
* --batchSize: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number * core_number.
* --partitionNum: the partition number, default is node_number * core_number.
