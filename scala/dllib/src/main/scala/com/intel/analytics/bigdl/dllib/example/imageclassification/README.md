## Summary
This example demonstrates how to use BigDL to load a BigDL or [Torch](http://torch.ch/) model trained on [ImageNet](http://image-net.org/) data, and then apply the loaded model to classify the contents of a set of images in Spark ML pipeline.

## Preparation

To start with this example, you need prepare your model and dataset.

1. Prepare model.

    The Torch ResNet model used in this example can be found in [Resnet Torch Model](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained).
    The BigDL Inception model used in this example can be trained with [BigDL Inception](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/inception).
    You can choose one of them, and then put the trained model in $modelPath, and set corresponding $modelType（torch or bigdl）.
   
2. Prepare predict dataset

    Put your image data for prediction in the ./predict folder. Alternatively, you may also use imagenet-2012 validation dataset to run the example, which can be found from <http://image-net.org/download-images>. After you download the file (ILSVRC2012_img_val.tar), run the follow commands to prepare the data.
    
    ```bash
    mkdir predict
    tar -xvf ILSVRC2012_img_val.tar -C ./predict/
    ```
  
  
     <code>Note: </code>For large dataset, you may want to read image data from HDFS.This command will transform the images into hadoop sequence files:

```bash
mkdir -p val/images
mv predict/* val/images/
java -cp bigdl_folder/lib/bigdl-VERSION-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator -f ./ --validationOnly --hasName
mv val/*.seq predict/
```

  
## Run this example

Command to run the example in Spark local mode:
```
spark-submit \
--master local[physcial_core_number] \
--driver-memory 10g --executor-memory 20g \
--class com.intel.analytics.bigdl.example.imageclassification.ImagePredictor \
./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--modelPath ./resnet-18.t7 \
--folder ./predict \
--modelType torch \
--batchSizePerCore 16 \
--isHdfs false
```
Command to run the example in Spark standalone mode:
```
spark-submit \
--master spark://... \
--executor-cores 8 \
--total-executor-cores 32 \
--class com.intel.analytics.bigdl.example.imageclassification.ImagePredictor \
./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--modelPath ./resnet-18.t7 \
--folder ./predict \
--modelType torch \
--batchSizePerCore 16 \
--isHdfs false
```
Command to run the example in Spark yarn mode:
```
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores 8 \
--num-executors 4 \
--class com.intel.analytics.bigdl.example.imageclassification.ImagePredictor \
./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--modelPath ./resnet-18.t7 \
--folder ./predict \
--modelType torch \
--batchSizePerCore 16 \
--isHdfs false
```
where 

* ```--modelPath``` is the path to the model file.
* ```--folder``` is the folder of predict images.
* ```--modelType``` is the type of model to load, it can be ```bigdl``` or ```torch```.
* ```--showNum``` is the result number to show, default 100.
* ```--batchSize``` is the batch size to use when do the prediction, default 32.
* ```--isHdfs``` is the type of predict data. "true" means reading sequence file from hdfs, "false" means reading local images, default "false". 
