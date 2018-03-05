## Overview
   Deep Learning Frames provides high-level APIs for scalable deep learning in Scala with Apache Spark.
   The current version of Deep Learning Frames provides a suite of tools around working with and processing images using deep learning. 
   Two exmaples demostrates how to use BigDL for transfer learning and applying deep learning models as scale.

## Model Inference
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
        Download [caffe inception v1](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel) and [deploy.proxfile](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt) from 
        and then put the trained model in $modelPath, and set corresponding $caffeDefPath.
   
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
                --class com.intel.analytics.bigdl.example.DLFrames.ModelInference \
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
                --class com.intel.analytics.bigdl.example.imageclassification.ImagePredictor \
                ./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
                --modelPath ./model/bvlc_googlenet.caffemodel \
                --folder ./predict \
                --batchSizePerCore 16 \
                --nEpochs 10
        ```
## Transfer Learning 
   1. DLFrames provides utilities to perform transfer learning on images, which is one of the fastest (code and run-time-wise) ways to start using deep learning.             
         
             val imagesDF: DataFrame = loadImages(params.folder, params.batchSize, spark.sqlContext)
       
             val Array(validationDF, trainingDF) = imagesDF.randomSplit(Array(0.2, 0.8), seed = 1)
       
             val criterion = ClassNLLCriterion[Float]()
       
             val loaded: AbstractModule[Activity, Activity, Float] = Module.loadCaffeModel[Float](params.caffeDefPath, params.modelPath)
             val model = Sequential[Float]().add(loaded)
               .add(Linear[Float](1000, 2)).add(ReLU()).add(LogSoftMax())
       
             val dlc: DLClassifier[Float] = new DLClassifier[Float](model, criterion, Array(3, 224, 224))
               .setBatchSize(4)
               .setOptimMethod(new Adam())
               .setLearningRate(1e-2)
               .setLearningRateDecay(1e-5)
               .setMaxEpoch(10)
               .setFeaturesCol("features")
               .setLabelCol("label")
               .setPredictionCol("prediction")
          
             val dlModel: DLModel[Float] = dlc.fit(trainingDF)
          
             val predictions = dlModel.setBatchSize(2).transform(validationDF)
       
             predictions.show(5)
         
   2. You can run the full TransferLearning example by following steps.
        
        2.1 Prepare pre-trained model and defenition file as Model Inference.
        
        2.2 Prepare dataset
        Put your image data for training and validation in the ./data folder. Alternatively, you may also use kaggle [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) train dataset to run the example. After you download the file (train.zip), run the follow commands to prepare the data.
    
            ```
                bash
                mkdir data
                unzip -xvf train.tar -C ./data/
            ```
  
        2.3 Run this example

        Command to run the example in Spark local mode:
        ```
                spark-submit \
                --master local[physcial_core_number] \
                --driver-memory 10g --executor-memory 20g \
                --class com.intel.analytics.bigdl.example.DLFrames.TransferLearning \
                ./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
                --modelPath ./model/bvlc_googlenet.caffemodel \
                --caffeDefPath ./model/deploy.prototxt \
                --batchSize 32 \
                --folder ./data \
                --nEpochs 10                
        ```

        Command to run the example in Spark yarn mode(TODO):
        ```
                spark-submit \
                --master yarn \
                --deploy-mode client \
                --executor-cores 8 \
                --num-executors 4 \
                --class com.intel.analytics.bigdl.example.imageclassification.TransferLearning \
                ./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
                --modelPath ./model/bvlc_googlenet.caffemodel \
                --folder ./data \
                --batchSize 32 \
                --nEpochs 10                
        ```