## Overview
   Deep Learning Frames provides high-level APIs for scalable deep learning in Scala with Apache Spark.
   The current version of Deep Learning Frames provides a suite of tools around working with and processing images using deep learning. 
   this exmaple demostrates how to use BigDL for transfer learning.

## Transfer Learning 
   1. DLFrames provides utilities to perform transfer learning on images, which is one of the fastest (code and run-time-wise) ways to start using deep learning.             
         
          val createLabel = udf((name: String) => if (name.contains("cat")) 1.0 else 0.0)
          val imagesDF: DataFrame = Utils.loadImages(params.folder, params.batchSize, spark.sqlContext)
            .withColumn("label", createLabel(col("imageName")))
            .withColumnRenamed("features", "imageFeatures")
    
          val Array(validationDF, trainingDF) = imagesDF.randomSplit(Array(0.90, 0.10), seed = 1L)
    
          val criterion = ClassNLLCriterion[Float]()
    
          val loadedModel: AbstractModule[Activity, Activity, Float] = Module
            .loadCaffeModel[Float](params.caffeDefPath, params.modelPath)
    
          val featurizer = new DLModel[Float](loadedModel, Array(3, 224, 224))
            .setBatchSize(params.batchSize)
            .setFeaturesCol("imageFeatures")
            .setPredictionCol("tmp1")
    
    
          val schemaTransformer = new Array2Vector()
            .setFeaturesCol("tmp1")
            .setPredictionCol("features")
    
          val lr = new LogisticRegression()
            .setMaxIter(20)
            .setRegParam(0.05)
            .setElasticNetParam(0.3)
            .setFeaturesCol("features")
    
          val pipeline = new Pipeline().setStages(
            Array(featurizer, schemaTransformer, lr))
    
          val pipelineModel = pipeline.fit(trainingDF)
    
          val predictions = pipelineModel.transform(trainingDF)
         
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
                --class com.intel.analytics.bigdl.example.DLFrames.ImageTransferLearning \
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
                --class com.intel.analytics.bigdl.example.imageclassification.ImageTransferLearning \
                ./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
                --modelPath ./model/bvlc_googlenet.caffemodel \
                --folder ./data \
                --batchSize 32 \
                --nEpochs 10                
        ```