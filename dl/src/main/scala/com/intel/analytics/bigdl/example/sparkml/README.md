## Summary
 This example shows how to predict ImageNet images on trained model.
1. Only support bigdl model and torch model, not support caffe model
2. Only support model of imagenet.
3. Only support imagenet images or similar.

## data
All predict images should be imagenet images, and put in the same folder.

## Steps to run this example:
1. Prepare predict data
2. Run the commands:

    * Spark local:
        ```bash
        ./dist/bin/bigdl.sh -- java -Xmx20g -Dspark.master="local[*]" -cp bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar  com.intel.analytics.bigdl.example.sparkml.ImageClassifier
                                     --modelPath $modelPath --folder $folder --modelType $modelType -c 4 -n 4
        ```

    * Spark cluster:
        ```bash
        ./dist/bin/bigdl.sh -- spark-submit --master your master address --driver-memory 10g --executor-memory 20g --class com.intel.analytics.bigdl.example.sparkml.ImageClassifier  bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar  --coreNum 8 --nodeNum 4 --partitionNum 4 --batchSize 32  --baseDir ./postClassification/ --env spark
        ```
