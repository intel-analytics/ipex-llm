## Overview
   Deep Learning Frames provides high-level APIs for scalable deep learning in Scala with Apache Spark.
   The current version of Deep Learning Frames provides a suite of tools around working with and processing images using deep learning. 
   Two exmaples demostrates how to use BigDL for transfer learning and applying deep learning models as scale.

## Model Inference
   1. You apply your own or known popular models to image data to make predictions or transform them into features.
            
            val imagesDF = loadImages(param.folder, param.batchSize, spark.sqlContext)
               val model = Module.loadCaffeModel[Float](param.caffeDefPath, param.modelPath)
               val dlmodel: DLModel[Float] = new DLClassifierModel[Float](
                 model, Array(3, 224, 224))
                 .setBatchSize(param.batchSize)
                 .setFeaturesCol("features")
                 .setPredictionCol("predict")
           
               val tranDF = dlmodel.transform(imagesDF)
           
               tranDF.select("predict", "imageName").show(5)
   
   2. You can also run the full ModelInference example following steps.
        2.1 Prepare pre-trained model and defenition file.
        Download [caffe inception v1](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel) and [deploy.proxfile](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt) from 
        and then put the trained model in $modelPath, and set corresponding $caffeDefPath.
   
        2.2 Prepare predict dataset
        Put your image data for prediction in the ./predict folder. Alternatively, you may also use imagenet-2012 validation dataset to run the example, which can be found from <http://image-net.org/download-images>. After you download the file (ILSVRC2012_img_val.tar), run the follow commands to prepare the data.
    
            ```bash
            mkdir predict
             tar -xvf ILSVRC2012_img_val.tar -C ./folder/
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
--class com.intel.analytics.bigdl.example.DLFrames.ModelInference \
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
