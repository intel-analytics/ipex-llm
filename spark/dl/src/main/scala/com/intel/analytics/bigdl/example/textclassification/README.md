## Summary
 This example use a (pre-trained GloVe embedding) to convert word to vector,
 and uses it to train the text classification model on a 20 Newsgroup dataset
 with 20 different categories. This model can achieve around 90% accuracy after 2 epochs training.
(It was first described in: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
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
    wget http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz
    tar zxf 20news-18828.tar.gz
    ```

3.  Put those data under BASE_DIR, and the final structure would look like this:

    ```
    [~/textclassification]$ tree . -L 1
    .
    ├── 20news-18828
    └── glove.6B
    ```

4.  Run the commands:
    * Spark local:
      * Execute:

        ```shell
        BASE_DIR=${PWD} # where is the data
        spark-submit --master "local[physical_core_number]" --driver-memory 20g                     \
                   --class com.intel.analytics.bigdl.example.textclassification.TextClassifier \
                   bigdl-VERSION-jar-with-dependencies.jar --batchSize 128              \
                   --baseDir ${BASE_DIR} --partitionNum 4
        ```

    * Spark cluster:
      * Standalone execute:

        ```shell
        MASTER=xxx.xxx.xxx.xxx:xxxx
        BASE_DIR=${PWD} # where is the data
        spark-submit --master ${MASTER} --driver-memory 20g --executor-memory 20g  \
                   --total-executor-cores 32 --executor-cores 8                                \
                   --class com.intel.analytics.bigdl.example.textclassification.TextClassifier \
                   bigdl-VERSION-jar-with-dependencies.jar --batchSize 128              \
                   --baseDir ${BASE_DIR} --partitionNum 32
        ```
        * Yarn client execute:
        
                ```shell
                BASE_DIR=${PWD} # where is the data
                spark-submit --master yarn --driver-memory 20g --executor-memory 20g  \
                           --num-executor 4 --executor-cores 8                                \
                           --class com.intel.analytics.bigdl.example.textclassification.TextClassifier \
                           bigdl-VERSION-jar-with-dependencies.jar --batchSize 128              \
                           --baseDir ${BASE_DIR} --partitionNum 32
                ```

      * NOTE: The total batch is: 128 and the batch per node is 128/nodeNum

4. Verify:
   * Search accuracy from log:
   ``` 
   [Epoch 1 0/15964][Iteration 1][Wall Clock 0.0s]
   
   top1 accuracy is Accuracy(correct: 14749, count: 15964, accuracy: 0.9238912
      553244801)
