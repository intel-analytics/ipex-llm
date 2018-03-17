## Overview
   Deep Learning Frames provides high-level APIs for scalable deep learning in Scala with Apache Spark.
   The current version of Deep Learning Frames provides a suite of tools around working with and processing images using deep learning. 
   This exmaple demostrates how to use BigDL to apply popular iamge deep learning models at scale.

## Image Model Inference
   1. You can apply your own or known popular models to image data to make predictions or transform them into features.
            
            val imagesDF = loadImages(param.folder, param.batchSize, spark.sqlContext)
               val model = Module.loadCaffeModel[Float](param.caffeDefPath, param.modelPath)
               val dlmodel: DLModel[Float] = new DLClassifierModel[Float](
                 model, Array(3, 224, 224))
                 .setBatchSize(param.batchSize)
                 .setFeaturesCol("features")
                 .setPredictionCol("predict")
           
               val tranDF = dlmodel.transform(imagesDF)
           
               tranDF.select("predict", "imageName").show(5)
   
   2. You can run the full ModelInference example by following steps.
        
        2.1 Prepare pre-trained model and defenition file.
        Download [caffe inception v1](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel) and [deploy.proxfile](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt)  
        then put the trained model in $modelPath, and set corresponding $caffeDefPath.
   
        2.2 Prepare predict dataset
        Put your image data for prediction in the ./predict folder. Alternatively, you may also use imagenet-2012 validation dataset to run the example, which can be found from <http://image-net.org/download-images>. After you download the file (ILSVRC2012_img_val.tar), run the follow commands to prepare the data.
    
            ```bash
            mkdir predict
             tar -xvf ILSVRC2012_img_val.tar -C ./folder/
            ```
  
        2.3 Run this example

        Command to run the example in Spark local mode:
        ```
                spark-submit \
                --master local[physcial_core_number] \
                --driver-memory 10g --executor-memory 20g \
                --class com.intel.analytics.bigdl.example.DLFrames.ImageInference \
                ./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
                --modelPath ./model/bvlc_googlenet.caffemodel \
                --caffeDefPath ./model/deploy.prototxt \
                --batchSize 32 \
                --folder ./predict \
                --nEpochs 10
                
        ```

        Command to run the example in Spark yarn mode(TODO):
        ```
                spark-submit \
                --master yarn \
                --deploy-mode client \
                --executor-cores 8 \
                --num-executors 4 \
                --class com.intel.analytics.bigdl.example.DLFrames.ImageInference \
                ./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
                --modelPath ./model/bvlc_googlenet.caffemodel \
                --caffeDefPath ./model/deploy.prototxt \
                --batchSize 32 \
                --folder ./predict \
                --nEpochs 10
        ```