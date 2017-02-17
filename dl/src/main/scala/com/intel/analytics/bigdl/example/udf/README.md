## Summary
 This example is to show how to create a user defined function to do the text classification with BigDL in Dataframe and Spark SQL.
 First use a (pre-trained GloVe embedding) to convert word to vector,
 and uses it to train the text classification model on a 20 Newsgroup dataset
 with 20 different categories. This model can achieve around 90% accuracy after 2 epochs training.
(It was first described in: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
Then create a UDF to do the text classification with this model, and use this UDF in Dataframe and Spark SQL.
## Data
* Embedding: 100-dimensional pre-trained GloVe embeddings of 400k words which trained on a 2014 dump of English Wikipedia.
* Training data: "20 Newsgroup dataset" which containing 20 categories and with totally 19997 texts.

## Steps to run this example:
1.  Download [Pre-train GloVe word embeddings](http://nlp.stanford.edu/data/glove.6B.zip)

    ```shell
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip -q glove.6B.zip -d glove.6B
    ```

2.  Download [20 Newsgroup dataset](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html) as the training data

    ```shell
    wget http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz
    tar zxf news20.tar.gz
    ```

3.  Put those data under BASE_DIR, and the final structure would look like this:

    ```
    [~/textclassification]$ tree . -L 1
    .
    ├── 20_newsgroup
    └── glove.6B
    ```

4.  Run the commands:
    * bigdl.sh would setup the essential environment for you and it would accept a spark-submit command as an input parameter.
    * Spark local:
      
      If you want to save the trained model ,
      * Execute:

        ```shell
        BASE_DIR=${PWD} # where is the data
        ./bigdl.sh -- spark-submit --master "local[*]" --driver-memory 20g                     \
                   --class com.intel.analytics.bigdl.example.udf.TextClassifierUDF \
                   bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --batchSize 128              \
                   --baseDir ${BASE_DIR} --partitionNum 4 --checkpoint ~/model/text
        ```
        
      If you have saved model ,
      * Execute:
     
             ```shell
             BASE_DIR=${PWD} # where is the data
             ./bigdl.sh -- spark-submit --master "local[*]" --driver-memory 20g                     \
                        --class com.intel.analytics.bigdl.example.udf.TextClassifierUDF \
                        bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --batchSize 128              \
                        --baseDir ${BASE_DIR} --partitionNum 4 --modelPath ~/model/text/model.1
             ```   
    * Spark cluster:
      * Execute:

        ```shell
        MASTER=xxx.xxx.xxx.xxx:xxxx
        BASE_DIR=${PWD} # where is the data
        ./bigdl.sh -- spark-submit --master ${MASTER} --driver-memory 5g --executor-memory 5g  \
                   --total-executor-cores 32 --executor-cores 8                                \
                   --class com.intel.analytics.bigdl.example.textclassification.TextClassifier \
                   bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --coreNum 8                  \
                   --nodeNum 4 --batchSize 32  --baseDir ${BASE_DIR} --partitionNum 32
        ```

      * NOTE: The total batch is: 32 * 4 as we specify nodeNum to be 4

5. Verify:
   * Show the prediction with UDF and original label in dataframe:
    
    df.withColumn("textType", classiferUDF($"features"))
                 .select("fileName", "textType", "label").show()
   ``` 
   +--------+--------+-----+
   |fileName|textType|label|
   +--------+--------+-----+
   |   60560|       4|    4|
   |   53071|       1|    1|
   |   60580|       4|    4|
   |    9963|       3|    3|
   |   53096|       1|    1|
   |   50441|       5|    5|
   |   60285|       4|    4|
   |   53409|       1|    1|
   |    9579|       3|    3|
   |   61133|       4|    4|
   +--------+--------+-----+
   ```
   Note: "textType" column is the prediction for the text, "label" column is the original label for the text. The prediction accuracy is above 95%.
   
   * Filter textType with UDF in dataframe:
       
       df.withColumn("textType", classiferUDF($"features"))
               .filter("textType = 1")
               .select("fileName", "textType", "label").show()
      ``` 
      +--------+--------+-----+
      |fileName|textType|label|
      +--------+--------+-----+
      |   51162|       1|    1|
      |   54248|       1|    1|
      |   51259|       1|    1|
      |   51125|       1|    1|
      |   53334|       1|    1|
      |   53628|       1|    1|
      |   53670|       1|    1|
      |   51171|       1|    1|
      |   53756|       1|    1|
      |   51158|       1|    1|
      +--------+--------+-----+
      ```
      
   * Show the prediction with UDF and orginal label in Spark SQL:
       
       sqlContext.sql("select fileName, textClassifier(featuress) as textType, label from textTable").show()
      ``` 
      +--------+--------+-----+
      |fileName|textType|label|
      +--------+--------+-----+
      |   51144|       1|    1|
      |   38329|       2|    2|
      |   51790|       5|    5|
      |   51609|       5|    5|
      |   52060|       5|    5|
      |    9811|       3|    3|
      |   53108|       1|    1|
      |   54226|       1|    1|
      |   52346|       5|    5|
      |   52113|       5|    5|
      +--------+--------+-----+
      ```
    * Filter textType with UDF in Spark SQL:
    
        sqlContext.sql("select fileName, textClassifier(features) as textType, label from textTable where textClassifier(features) = 1").show()
       
       ```
            +--------+--------+-----+
            |fileName|textType|label|
            +--------+--------+-----+
            |   53117|       1|    1|
            |   53160|       1|    1|
            |   51178|       1|    1|
            |   53386|       1|    1|
            |   51165|       1|    1|
            |   53284|       1|    1|
            |   53423|       1|    1|
            |   51129|       1|    1|
            |   53498|       1|    1|
            |   54143|       1|    1|
            +--------+--------+-----+
       ```