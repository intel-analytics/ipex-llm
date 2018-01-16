## Summary
 This example is to show how to load BigDL model as UDF to perform predictions in Spark SQL/Dataframes.
 
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
                                                              [Build Page](https://bigdl-project.github.io/master/#ScalaUserGuide/install-build-src/).


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
   MASTER=local[*] # the master url
   spark-submit --master $MASTER --driver-memory 20g \
       --class com.intel.analytics.bigdl.example.udfpredictor.DataframePredictor \
         ./dist/lib/bigdl-$VERSION-jar-with-dependencies.jar \
       --batchSize 32 \
       --baseDir $BASE_DIR \
       --partitionNum 4 \
       --checkpoint  $BASE_DIR/model/text \
       --dataDir $BASE_DIR/test
   ```
        
   In the above commands, 
   
   --batchSize: how many text files to be trained at one time
   
   --baseDir: folder containing trainning text files and word2Vec embedding.
   
   --partitionNum: number to partition training data
   
   --checkpoint: location to save model
   
   --dataDir: Directory containing the test data

   If you are running spark cluster mode, you also need to set --executor-cores and --total-executor-cores, and the
   --batchSize should be a multiple of node_number*core_number.

   Example:

   ```shell
   BASE_DIR=${PWD} # where is the data
   MASTER=xxx.xxx.xxx.xxx:xxxx # the master url
   spark-submit --master $MASTER --driver-memory 20g \
       --executor-cores 8 \
       --total-executor-cores 32 \
       --class com.intel.analytics.bigdl.example.udfpredictor.DataframePredictor \
         ./dist/lib/bigdl-$VERSION-jar-with-dependencies.jar \
       --batchSize 32 \
       --baseDir $BASE_DIR \
       --partitionNum 4 \
       --checkpoint  $BASE_DIR/model/text \
       --dataDir $BASE_DIR/test
   ```
   
   If you have saved model, you need to use this command to predict text classification of text records with UDF.
      
   Example:
     
   ```shell
   BASE_DIR=${PWD} # where is the data
   MASTER=local[*] # the master url
   spark-submit --master $MASTER --driver-memory 5g \
              --class com.intel.analytics.bigdl.example.udfpredictor.DataframePredictor \
                bigdl-$VERSION-jar-with-dependencies.jar \
              --baseDir $BASE_DIR \
              --modelPath  $BASE_DIR/model/text/model.1 \
              --dataDir $BASE_DIR/test
   ```
   In the above commands, 
   
   -baseDir: folder containing trainning text files and word2Vec embedding 
   
   --modelPath: model location
   
   --dataDir: Directory containing the test data
        
* Verification
        
    * Show the predicted label with UDF for text records:

    ``` 
    val classifyDF1 = df.withColumn("textLabel", classifierUDF($"text"))
        .select("filename", "text", "textLabel")
    classifyDF1.show() 
    +--------+--------------------+---------+
    |filename|                text|textLabel|
    +--------+--------------------+---------+
    |   10014|Xref: cantaloupe....|        3|
    |  102615|Newsgroups: rec.s...|       10|
    |  102642|Newsgroups: rec.s...|       10|
    |  102685|Path: cantaloupe....|       10|
    |  102741|Newsgroups: rec.a...|        8|
    |  102771|Xref: cantaloupe....|        8|
    |  102826|Newsgroups: rec.a...|        8|
    |  102970|Newsgroups: rec.a...|        8|
    |  102982|Newsgroups: rec.a...|        8|
    |  103125|Newsgroups: rec.a...|        8|
    |  103329|Path: cantaloupe....|        8|
    |  103497|Path: cantaloupe....|        8|
    |  103515|Path: cantaloupe....|        8|
    |  103781|Xref: cantaloupe....|        8|
    |  104333|Newsgroups: rec.m...|        9|
    |  104341|Path: cantaloupe....|        9|
    |  104381|Newsgroups: rec.m...|        9|
    |  104509|Newsgroups: rec.m...|        9|
    |  104542|Xref: cantaloupe....|        9|
    |  104590|Newsgroups: rec.s...|       10|
    +--------+--------------------+---------+
    ```
         
    Note: "textLabel" column is the prediction for the text.
         
     * Filter text label with UDF:
             
      ```
      val filteredDF1 = df.filter(classifierUDF($"text") === 9)
      filteredDF1.show()
      +--------+--------------------+
      |filename|                text|
      +--------+--------------------+
      |  104333|Newsgroups: rec.m...|
      |  104341|Path: cantaloupe....|
      |  104381|Newsgroups: rec.m...|
      |  104509|Newsgroups: rec.m...|
      |  104542|Xref: cantaloupe....|
      |  104595|Newsgroups: rec.m...|
      |  104753|Path: cantaloupe....|
      |  104806|Newsgroups: rec.m...|
      +--------+--------------------+
      ```
            
     * Join the text type table to show the text type name :
             
     ``` 
     val df_join = classifyDF1.join(types, "textLabel")
     df_join.show()
     +--------+--------------------+---------+-------------+
     |filename|                text|textLabel|     textType|
     +--------+--------------------+---------+-------------+
     |   51141|Path: cantaloupe....|        1|  alt.atheism|
     |   51189|Newsgroups: alt.a...|        1|  alt.atheism|
     |   51202|Newsgroups: alt.a...|        1|  alt.atheism|
     |   51313|Newsgroups: alt.a...|        1|  alt.atheism|
     |   53165|Path: cantaloupe....|        1|  alt.atheism|
     |   53237|Path: cantaloupe....|        1|  alt.atheism|
     |   53252|Path: cantaloupe....|        1|  alt.atheism|
     |   53383|Path: cantaloupe....|        1|  alt.atheism|
     |   53577|Xref: cantaloupe....|        1|  alt.atheism|
     |   84114|Xref: cantaloupe....|        1|  alt.atheism|
     |   15244|Xref: cantaloupe....|        2|comp.graphics|
     |   38265|Newsgroups: comp....|        2|comp.graphics|
     |   38330|Path: cantaloupe....|        2|comp.graphics|
     |   38363|Xref: cantaloupe....|        2|comp.graphics|
     |   38600|Xref: cantaloupe....|        2|comp.graphics|
     |   38684|Newsgroups: comp....|        2|comp.graphics|
     |   38766|Newsgroups: comp....|        2|comp.graphics|
     |   38865|Path: cantaloupe....|        2|comp.graphics|
     |   38958|Newsgroups: comp....|        2|comp.graphics|
     |   38999|Path: cantaloupe....|        2|comp.graphics|
     +--------+--------------------+---------+-------------+
     ```
          
    * Do the aggregation of data frame with predicted text label:    
                
     ``` 
     val typeCount = classifyDF1.groupBy($"textLabel").count()
     typeCount.show()
                         
     +---------+-----+
     |        1|   10|
     |        2|   11|
     |        3|   11|
     |        4|   10|
     |        5|   10|
     |        6|    9|
     |        7|   11|
     |        8|   10|
     |        9|   10|
     |       10|    9|
     |       11|   10|
     |       12|    9|
     |       13|   11|
     |       14|   10|
     |       15|   10|
     |       16|   10|
     |       17|   10|
     |       18|   11|
     |       19|   13|
     |       20|    5|
     +---------+-----+
     ```
          
     * Show the predicted label with UDF in Spark SQL:
             
     ```
     val classifyDF2 = spark
       .sql("SELECT filename, textClassifier(text) AS textType_sql, text " +
         "FROM textTable")
       classifyDF2.show()
                             
       +--------+------------+--------------------+
       |filename|textType_sql|                text|
       +--------+------------+--------------------+
       |   10014|           3|Xref: cantaloupe....|
       |  102615|          10|Newsgroups: rec.s...|
       |  102642|          10|Newsgroups: rec.s...|
       |  102685|          10|Path: cantaloupe....|
       |  102741|           8|Newsgroups: rec.a...|
       |  102771|           8|Xref: cantaloupe....|
       |  102826|           8|Newsgroups: rec.a...|
       |  102970|           8|Newsgroups: rec.a...|
       |  102982|           8|Newsgroups: rec.a...|
       |  103125|           8|Newsgroups: rec.a...|
       |  103329|           8|Path: cantaloupe....|
       |  103497|           8|Path: cantaloupe....|
       |  103515|           8|Path: cantaloupe....|
       |  103781|           8|Xref: cantaloupe....|
       |  104333|           9|Newsgroups: rec.m...|
       |  104341|           9|Path: cantaloupe....|
       |  104381|           9|Newsgroups: rec.m...|
       |  104509|           9|Newsgroups: rec.m...|
       |  104542|           9|Xref: cantaloupe....|
       |  104590|          10|Newsgroups: rec.s...|
       +--------+------------+--------------------+
     ```
             
     * Filter text label with UDF for incoming stream in Spark SQL:
           
     ```
     val filteredDF2 = spark
        .sql("SELECT filename, textClassifier(text) AS textType_sql, text " +
          "FROM textTable WHERE textClassifier(text) = 9")
     filteredDF2.show()
     +--------+------------+--------------------+
     |filename|textType_sql|                text|
     +--------+------------+--------------------+
     |  104333|           9|Newsgroups: rec.m...|
     |  104341|           9|Path: cantaloupe....|
     |  104381|           9|Newsgroups: rec.m...|
     |  104509|           9|Newsgroups: rec.m...|
     |  104542|           9|Xref: cantaloupe....|
     |  104595|           9|Newsgroups: rec.m...|
     |  104753|           9|Path: cantaloupe....|
     |  104806|           9|Newsgroups: rec.m...|
     +--------+------------+--------------------+
     ```