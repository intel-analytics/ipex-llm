## Summary
 This example is to show how to create a user defined function to do the text classification with BigDL, and use this UDF in structured streaming.
 
 First use a (pre-trained GloVe embedding) to convert word to vector,
 and uses it to train the text classification model on a 20 Newsgroup dataset
 with 20 different categories. This model can achieve around 90% accuracy after 2 epochs training.
(It was first described in: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)

Then create a UDF to do the text classification with this model, and use this UDF to do the prediction in Structured Streaming.
## Data
* Embedding: 100-dimensional pre-trained GloVe embeddings of 400k words which trained on a 2014 dump of English Wikipedia.
* Training data: "20 Newsgroup dataset" which containing 20 categories and with totally 19997 texts.

## Get the JAR
Since this example use Kafka source for structured streaming , please build the source code with spark_2.1 profile:

   ```
   bash make-dist.sh -P spark_2.1
   ```
  
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

4. Run this command to generate test data for prediction in Structured Streaming:
    ```bash
    usage: create_test_texts.py [-h] [-s SRC] [-t TEST]
    
    Create test text files from source
    
    optional arguments:
     -h, --help            show this help message and exit
     -s SRC, --src SRC     source directory
     -t TEST, --test TEST  test directory
     
    Example: python ./create_test_texts.py -s BASE_DIR/20_newsgroup -t BASE_DIR/test
       
    ```  
5. Run this command to publish text data to the Kafka topic:
    Example: 
    ```
    scala -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
        com.intel.analytics.bigdl.example.structuredStreamUdf.TextProducerKafka \
    -b  kafka_broker_list \
    -t topic \
    -f BASE_DIR/test \
    -i 10 \
    --batchsize 4
    ```
    In the above commands
    * -b: Kafa broker List
    * -t: target topic to publish
    * -f: folder containing text files to be published
    * -i: publish interval in second
    * --batchSize: how many text files to be published at one time

6. Start the consumer to subscribe the text streaming and do prediction with UDF.

    Run the commands:
    
    * Spark local:
      
      If you do not haved pre-trained model, you need to use this command to train the model and use this model to predict text classification of incoming text streaming with UDF.
      
        Example:

        ```shell
        BASE_DIR=${PWD} # where is the data
        ./bigdl.sh -- spark-submit --master "local[*]" --driver-memory 20g \
            --class com.intel.analytics.bigdl.example.structuredStreamUdf.TextClassifierConsumerKafka \
              bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
            -c 4 \
            -n 1 \
            --batchSize 32 \
            --baseDir BASE_DIR \
            --partitionNum 4 \
            --checkpoint  ~/model/text \
            --bootstrap localhost:9092
            --topic topic1
        ```
        
       In the above commands, 
        * -c: How many cores of your machine will be used in the training. Note that the core number should be physical core number. If your machine turn on hyper threading, one physical core will map to two OS core.
        * -n: Node number.
        * --batchSize: how many text files to be trained at one time
        * -baseDir: folder containing trainning text files.
        * --partitionNum: number to partition training data
        * --checkpoint: location to save model
        * --bootstrap: boot strap server to subscribe
        * --topic: topic to subscribe
    
      If you have saved model, you need to use this command to predict text classification of incoming text streaming with UDF.
      
        Example:
     
        ```shell
        BASE_DIR=${PWD} # where is the data
        ./bigdl.sh -- spark-submit --master "local[*]" --driver-memory 20g \
                   --class com.intel.analytics.bigdl.example.structuredStreamUdf.TextClassifierConsumerKafka \
                     bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
                   -c 4 \
                   -n 1 \  
                   --modelPath  ~/model/text/model.1 \
                   --bootstrap localhost:9092
                   --topic topic1
        ```
        In the above commands, 
        * -c: How many cores of your machine will be used in the training. Note that the core number should be physical core number. If your machine turn on hyper threading, one physical core will map to two OS core.
        * -n: Node number.
        * --partitionNum: 
        * --modelPath: model location
        * --bootstrap: boot strap server to subscribe
        * --topic: topic to subscribe
    
    * Spark cluster:
      * Execute:

        ```shell
        MASTER=xxx.xxx.xxx.xxx:xxxx
        BASE_DIR=${PWD} # where is the data
        ./bigdl.sh -- spark-submit --master ${MASTER} \
            --driver-memory 5g --executor-memory 5g  \
            --total-executor-cores 32 --executor-cores 8 \
            --class com.intel.analytics.bigdl.example.structuredStreamUdf.TextClassifierConsumerKafka \ 
            bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
            -c 8 \ -n 4 \
            --modelPath  ~/model/text/model.1 \
            --bootstrap localhost:9092 \
            --topic topic1
        ```
        In the above commands, 
         * -c: How many cores of your machine will be used in the training. Note that the core number should be physical core number. If your machine turn on hyper threading, one physical core will map to two OS core.
         * -n: Node number.
         * --partitionNum: 
         * --modelPath: model location
         * --bootstrap: boot strap server to subscribe
         * --topic: topic to subscribe
      

7. Verify:
   * Show the predicted label with UDF for incoming streaming:
      
   ``` 
   val classifyDF1 = df.withColumn("textLabel", classiferUDF($"text"))
               .select("fileName", "textLabel", "text")
   val classifyQuery1 = classifyDF1.writeStream
               .format("console")
               .start()  
   +--------+--------------------+---------+
   |fileName|                text|textLabel|
   +--------+--------------------+---------+
   |  100521|Path: cantaloupe....|       10|
   |  101551|Path: cantaloupe....|        8|
   |  101552|Newsgroups: rec.a...|        8|
   |  101553|Xref: cantaloupe....|        8|
   +--------+--------------------+---------+
   ```
   Note: "textLabel" column is the prediction for the text.
   
   * Filter text label with UDF in stream:
       
      ```
      val filteredDF1 = df.filter(classiferUDF($"text") === 8)
      val filteredQuery1 = filteredDF1.writeStream
                     .format("console")
                     .start()
      +--------------------+--------+
      |                text|filename|
      +--------------------+--------+
      |Path: cantaloupe....|  101551|
      |Newsgroups: rec.a...|  101552|
      |Xref: cantaloupe....|  101553|
      +--------------------+--------+
      ```
      
   * Join the static text type table with stream to show the text type name :
       
      ``` 
      val df_join = classifyDF1.join(types, "textLabel")
      val classifyQuery_join = df_join.writeStream
                     .format("console")
                     .start()
      +---------+--------+--------------------+------------------+
      |textLabel|fileName|                text|          textType|
      +---------+--------+--------------------+------------------+
      |       10|  100521|Path: cantaloupe....|rec.sport.baseball|
      |        8|  101551|Path: cantaloupe....|         rec.autos|
      |        8|  101552|Newsgroups: rec.a...|         rec.autos|
      |        8|  101553|Xref: cantaloupe....|         rec.autos|
      +---------+--------+--------------------+------------------+
      ```
    
   * Do the aggregation of stream with predicted text label:    
          
     ``` 
     val typeCount = classifyDF1.groupBy($"textLabel").count()
     val aggQuery = typeCount.writeStream
                   .outputMode("complete")
                   .format("console")
                   .start()
                   
      +---------+-----+
      |textLabel|count|
      +---------+-----+
      |        8|    3|
      |       10|    1|
      +---------+-----+
     ```
    
    * Show the predicted label with UDF for incoming stream in Spark SQL:
       
       ```
       val classifyDF2 = spark
                       .sql("SELECT fileName, textClassifier(text) AS textType_sql, text FROM textTable")
       val classifyQuery2 = classifyDF2.writeStream
                       .format("console")
                       .start()
                       
        +--------+------------+--------------------+
        |fileName|textType_sql|                text|
        +--------+------------+--------------------+
        |  101725|           9|Path: cantaloupe....|
        |  102151|          10|Path: cantaloupe....|
        |  102584|          10|Path: cantaloupe....|
        |  102585|          10|Newsgroups: rec.s...|
        +--------+------------+--------------------+
       ```
       
   * Filter text label with UDF for incoming stream in Spark SQL:
     
     ```
      val filteredDF2 = spark
                  .sql("SELECT fileName, textClassifier(text) AS textType_sql, text " +
                    "FROM textTable WHERE textClassifier(text) = 9")
      val filteredQuery2 = filteredDF2.writeStream
                  .format("console")
                  .start()
      +--------+------------+--------------------+
      |fileName|textType_sql|                text|
      +--------+------------+--------------------+
      |  101725|           9|Path: cantaloupe....|
      +--------+------------+--------------------+
      ```