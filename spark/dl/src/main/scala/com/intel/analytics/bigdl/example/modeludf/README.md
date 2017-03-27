## Summary
 This example is to show how to create a user defined function to do the text classification with BigDL, and use this UDF in Spark SQL (spark 1.5, spark 1.6) and structured streaming (spark 2.0+).
 
 First use a (pre-trained GloVe embedding) to convert word to vector,
 and uses it to train the text classification model on a 20 Newsgroup dataset
 with 20 different categories. This model can achieve around 90% accuracy after 2 epochs training.
(It was first described in: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)

Then create a UDF to do the text classification with this model, and use this UDF to do the prediction in Spark SQL (spark 1.5, spark 1.6) and Structured Streaming (spark 2.0).
## Data
* Embedding: 100-dimensional pre-trained GloVe embeddings of 400k words which trained on a 2014 dump of English Wikipedia.
* Training data: "20 Newsgroup dataset" which containing 20 categories and with totally 19997 texts.

## Get the JAR
Please build the source code with your specific version of spark referring the
                                                              [Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page).


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
    
5. Run Spark SQL example

* Training the model and make predictions.
   
   Run the commands:
   
   If you do not have the pre-trained model, you need to use this command to train the model and use this model to predict text classification of text records with UDF.

        Example:

        ```shell
        BASE_DIR=${PWD} # where is the data
        MASTER=loca[*] # the master url
        ./dist/bin/bigdl.sh -- spark-submit --master $MASTER --driver-memory 20g \
            --class com.intel.analytics.bigdl.example.modeludf.BatchPredictor \
              ./dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
            -c 4 \
            -n 1 \
            --batchSize 32 \
            --baseDir $BASE_DIR \
            --partitionNum 4 \
            --checkpoint  $BASE_DIR/model/text \
            --dataDir $BASE_DIR/test
        ```
        
       In the above commands, 
        * -c: How many cores of your machine will be used in the training. Note that the core number should be physical core number. If your machine turn on hyper threading, one physical core will map to two OS core.
        * -n: Node number.
        * --batchSize: how many text files to be trained at one time
        * -baseDir: folder containing trainning text files and word2Vec embedding.
        * --partitionNum: number to partition training data
        * --checkpoint: location to save model
        * --dataDir: Directory containing the test data

      If you have saved model, you need to use this command to predict text classification of text records with UDF.
      
        Example:
     
        ```shell
        BASE_DIR=${PWD} # where is the data
        MASTER=loca[*] # the master url
        ./bigdl.sh -- spark-submit --master $MASTER --driver-memory 5g \
                   --class com.intel.analytics.bigdl.example.modeludf.BatchPredictor \
                     bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
                   -c 4 \
                   -n 1 \  
                   --baseDir $BASE_DIR \
                   --modelPath  $BASE_DIR/model/text/model.1 \
                   --dataDir $BASE_DIR/test
        ```
        In the above commands, 
        * -c: How many cores of your machine will be used in the training. Note that the core number should be physical core number. If your machine turn on hyper threading, one physical core will map to two OS core.
        * -n: Node number.
        * -baseDir: folder containing trainning text files and word2Vec embedding 
        * --modelPath: model location
        * --dataDir: Directory containing the test data
        
* Show the predicted label with UDF for text records:

         ``` 
         val classifyDF1 = df.withColumn("textLabel", classifierUDF($"text"))
             .select("filename", "text", "textLabel").orderBy("filename")
         classifyDF1.show() 
         +--------+--------------------+---------+                                       
         |filename|                text|textLabel|
         +--------+--------------------+---------+
         |  101602|Xref: cantaloupe....|        8|
         |  101661|Xref: cantaloupe....|        8|
         |  101725|Path: cantaloupe....|        9|
         |   10191|Newsgroups: comp....|        3|
         |  102852|Path: cantaloupe....|        8|
         |  102903|Newsgroups: rec.a...|        8|
         |  103018|Path: cantaloupe....|        8|
         |  103031|Path: cantaloupe....|        8|
         |  103073|Path: cantaloupe....|        8|
         |  103167|Path: cantaloupe....|        9|
         |  103171|Xref: cantaloupe....|        9|
         |  103188|Path: cantaloupe....|        9|
         |  103264|Xref: cantaloupe....|        8|
         |  103335|Newsgroups: rec.a...|        8|
         |  103411|Newsgroups: rec.a...|        8|
         |  104375|Path: cantaloupe....|       10|
         |  104410|Newsgroups: rec.s...|       10|
         |  104503|Path: cantaloupe....|       10|
         |  104512|Newsgroups: rec.s...|       10|
         |  104557|Newsgroups: rec.m...|        9|
         +--------+--------------------+---------+
         ```
         Note: "textLabel" column is the prediction for the text.
         
         * Filter text label with UDF:
             
            ```
            val filteredDF1 = df.filter(classifierUDF($"text") === 8).orderBy("filename")
            filteredDF1.show()
            +--------+--------------------+
            |filename|                text|
            +--------+--------------------+
            |  101602|Xref: cantaloupe....|
            |  101661|Xref: cantaloupe....|
            |  103031|Path: cantaloupe....|
            |  103073|Path: cantaloupe....|
            |  103411|Newsgroups: rec.a...|
            |  178894|Xref: cantaloupe....|
            |  178899|Xref: cantaloupe....|
            |   20765|Path: cantaloupe....|
            |   53658|From: myers@hpfcs...|
            |   53712|Newsgroups: sci.e...|
            |   54091|Newsgroups: rec.s...|
            |   54135|Path: cantaloupe....|
            |   54454|Newsgroups: talk....|
            |   61478|Newsgroups: sci.s...|
            |   74766|Newsgroups: misc....|
            +--------+--------------------+
            ```
            
         * Join the text type table to show the text type name :
             
            ``` 
            val df_join = classifyDF1.join(types, "textLabel").orderBy("filename")
            df_join.show()
            +---------+--------+--------------------+--------------------+
            |        8|  101602|Xref: cantaloupe....|           rec.autos|
            |        8|  101661|Xref: cantaloupe....|           rec.autos|
            |        9|  101725|Path: cantaloupe....|     rec.motorcycles|
            |        3|   10191|Newsgroups: comp....|comp.os.ms-window...|
            |        8|  102852|Path: cantaloupe....|           rec.autos|
            |        8|  102903|Newsgroups: rec.a...|           rec.autos|
            |        8|  103018|Path: cantaloupe....|           rec.autos|
            |        8|  103031|Path: cantaloupe....|           rec.autos|
            |        8|  103073|Path: cantaloupe....|           rec.autos|
            |        9|  103167|Path: cantaloupe....|     rec.motorcycles|
            |        9|  103171|Xref: cantaloupe....|     rec.motorcycles|
            |        9|  103188|Path: cantaloupe....|     rec.motorcycles|
            |        8|  103264|Xref: cantaloupe....|           rec.autos|
            |        8|  103335|Newsgroups: rec.a...|           rec.autos|
            |        8|  103411|Newsgroups: rec.a...|           rec.autos|
            |       10|  104375|Path: cantaloupe....|  rec.sport.baseball|
            |       10|  104410|Newsgroups: rec.s...|  rec.sport.baseball|
            |       10|  104503|Path: cantaloupe....|  rec.sport.baseball|
            |       10|  104512|Newsgroups: rec.s...|  rec.sport.baseball|
            |        9|  104557|Newsgroups: rec.m...|     rec.motorcycles|
            +---------+--------+--------------------+--------------------+
            ```
          
         * Do the aggregation of stream with predicted text label:    
                
           ``` 
           val typeCount = classifyDF1.groupBy($"textLabel").count().orderBy("textLabel")
           typeCount.show()
                         
            +---------+-----+                                                               
            |textLabel|count|
            +---------+-----+
            |        1|    9|
            |        2|   10|
            |        3|   10|
            |        4|   10|
            |        5|   10|
            |        6|   10|
            |        7|   10|
            |        8|   10|
            |        9|   10|
            |       10|   10|
            |       11|   10|
            |       12|   11|
            |       13|   10|
            |       14|   10|
            |       15|   11|
            |       16|    9|
            |       17|    9|
            |       18|   10|
            |       19|    8|
            |       20|   13|
            +---------+-----+
           ```
          
          * Show the predicted label with UDF in Spark SQL:
             
             ```
             val classifyDF2 = spark
               .sql("SELECT filename, textClassifier(text) AS textType_sql, text " +
                 "FROM textTable order by filename")
             classifyDF2.show()
                             
              +--------+------------+--------------------+
              |filename|textType_sql|                text|
              +--------+------------+--------------------+
              |  101602|           8|Xref: cantaloupe....|
              |  101661|           8|Xref: cantaloupe....|
              |  101725|           9|Path: cantaloupe....|
              |   10191|           3|Newsgroups: comp....|
              |  102852|           8|Path: cantaloupe....|
              |  102903|           8|Newsgroups: rec.a...|
              |  103018|           8|Path: cantaloupe....|
              |  103031|           8|Path: cantaloupe....|
              |  103073|           8|Path: cantaloupe....|
              |  103167|           9|Path: cantaloupe....|
              |  103171|           9|Xref: cantaloupe....|
              |  103188|           9|Path: cantaloupe....|
              |  103264|           8|Xref: cantaloupe....|
              |  103335|           8|Newsgroups: rec.a...|
              |  103411|           8|Newsgroups: rec.a...|
              |  104375|          10|Path: cantaloupe....|
              |  104410|          10|Newsgroups: rec.s...|
              |  104503|          10|Path: cantaloupe....|
              |  104512|          10|Newsgroups: rec.s...|
              |  104557|           9|Newsgroups: rec.m...|
              +--------+------------+--------------------+
             ```
             
         * Filter text label with UDF for incoming stream in Spark SQL:
           
           ```
            val filteredDF2 = spark
               .sql("SELECT filename, textClassifier(text) AS textType_sql, text " +
                 "FROM textTable WHERE textClassifier(text) = 9 order by filename")
            filteredDF2.show()
            +--------+------------+--------------------+                                    
            |filename|textType_sql|                text|
            +--------+------------+--------------------+
            |  101725|           9|Path: cantaloupe....|
            |  103167|           9|Path: cantaloupe....|
            |  103171|           9|Xref: cantaloupe....|
            |  104557|           9|Newsgroups: rec.m...|
            |  104634|          14|Newsgroups: rec.m...|
            |  104684|           9|Path: cantaloupe....|
            |  105253|           9|Newsgroups: rec.m...|
            |   21565|           9|Xref: cantaloupe....|
            |   38516|           2|Path: cantaloupe....|
            |   38567|           2|Newsgroups: comp....|
            |   39002|          14|Path: cantaloupe....|
            |   54251|           9|Path: cantaloupe....|
            |   54273|           9|Newsgroups: sci.e...|
            |   55120|           9|Path: cantaloupe....|
            |   58128|          14|Path: cantaloupe....|
            |   58146|          18|Path: cantaloupe....|
            |   75373|          18|Path: cantaloupe....|
            |   75896|           7|Xref: cantaloupe....|
            |   76325|          18|Newsgroups: talk....|
            +--------+------------+--------------------+
            ```
   
5. Run Structured Streaming example

 > Note: To run this example, you spark version must be equal or higher than spark 2.0 

  * Start the consumer to subscribe the text streaming and do prediction with UDF.

   Run the commands:
          
   If you do not have the pre-trained model, you need to use this command to train the model and use this model to predict text classification of incoming text streaming with UDF.
      
        Example:

        ```shell
        BASE_DIR=${PWD} # where is the data
        MASTER=loca[*]  # the master url
        ./dist/bin/bigdl.sh -- spark-submit --master $MASTER --driver-memory 20g \
            --class com.intel.analytics.bigdl.example.modeludf.FileStreamingConsumer \
              ./dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
            -c 4 \
            -n 1 \
            --batchSize 32 \
            --baseDir $BASE_DIR \
            --partitionNum 4 \
            --checkpoint  $BASE_DIR/model/text \
            --dataDir $BASE_DIR/data/text/parquet
        ```
        
       In the above commands, 
        * -c: How many cores of your machine will be used in the training. Note that the core number should be physical core number. If your machine turn on hyper threading, one physical core will map to two OS core.
        * -n: Node number.
        * --batchSize: how many text files to be trained at one time
        * -baseDir: folder containing trainning text files and word2Vec embedding.
        * --partitionNum: number to partition training data
        * --checkpoint: location to save model
        * --dataDir: Directory to subscribe
    
   If you have saved model, you need to use this command to predict text classification of incoming text streaming with UDF.
      
        Example:
     
        ```shell
        BASE_DIR=${PWD} # where is the data, please modify it accordingly
        MASTER=loca[*] # the master url, please modify it accordingly
        ./bigdl.sh -- spark-submit --master MASTER --driver-memory 5g \
                   --class com.intel.analytics.bigdl.example.modeludf.FileStreamingConsumer \
                     bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
                   -c 4 \
                   -n 1 \  
                   --baseDir $BASE_DIR \
                   --modelPath  $BASE_DIR/model/text/model.1 \
                   --subDir $BASE_DIR/data/text/parquet
        ```
        In the above commands, 
        * -c: How many cores of your machine will be used in the training. Note that the core number should be physical core number. If your machine turn on hyper threading, one physical core will map to two OS core.
        * -n: Node number.
        * -baseDir: folder containing trainning text files and word2Vec embedding 
        * --modelPath: model location
        * --dataDir: Directory to subscribe
    
* Run this command to publish text data to the target directory:
    Example: 
    ```
    spark-submit --master "local[*]" \
        --class com.intel.analytics.bigdl.example.modeludf.FileStreamingProducer \
        dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
        -s $BASE_DIR/test \
        -d $BASE_DIR/data/text/parquet \
        -b 4 \
        -i 5
    ```
    In the above commands
    * -s: source folder containing text files to  to be published
    * -d: target directory to be published to
    * -i: publish interval in second
    * -b: how many text files to be published at one time      

* Show the predicted label with UDF for text records:
      
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