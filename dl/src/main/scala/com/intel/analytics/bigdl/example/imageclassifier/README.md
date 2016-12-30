## Summary
 This example shows how to predict images on trained model.
1. Only support bigdl model and torch model, not support caffe model
2. Only support model of imagenet.
3. Only support imagenet images or similar.

## data
All predict images should be put in the same folder.

## Steps to run this example:
1. Prepare predict data
2. Run environment script
   ```bash
   source ./scripts/bigdl.sh
   ```
3. Run the commands:

    * Spark local:
        ```bash
        java -Xmx20g -Dspark.master="local[*]" -cp bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar:$spark-jar  com.intel.analytics.bigdl.example.imageclassifier.DLClassifier
         --modelPath $modelPath --folder $folder --modelType $modelType -c $core -n $nodeNumber
        ```

    * Spark cluster:
        ```bash
        spark-submit --driver-memory 10g --executor-memory 20g --class com.intel.analytics.bigdl.example.imageclassifier.DLClassifier bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar
        --modelPath $modelPath --folder $folder --modelType $modelType -c $core -n $nodeNumber --batchSize 32
        ```