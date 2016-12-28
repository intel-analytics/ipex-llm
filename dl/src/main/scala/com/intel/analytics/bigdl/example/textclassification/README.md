## Summary
 This example use a (pre-trained GloVe embedding) to convert word to vector,
 and uses it to train a text classification model on the 20 Newsgroup dataset
 with 20 different categories. This model can achieve around 90% accuracy after 2 epoches training.
(It was first described in: https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py)
## Data
* Embedding: 100-dimensional pre-trained GloVe embeddings of 400k words which trained on a 2014 dump of English Wikipedia.
* Training data: "20 Newsgroup dataset" which containing 20 categories and with totally 19997 texts.

## Steps to run this example:
1. Download [Pre-train GloVe word embeddings](http://nlp.stanford.edu/projects/glove/)
2. Download [20 Newsgroup dataset](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html) as the training data
3. Run the commands:

    * No Spark:
      * bigdl.sh
      * java -Xmx10g -cp bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar  com.intel.analytics.bigdl.example.textclassification.TextClassifier  --baseDir $BASE_DIR  --batchSize 128 --nospark
      * (NB: BASE_DIR is the root directory which containing the embedding and the training data. There are other optional parameters you can use as well. i.e: batchSize, trainingSplit)

    * Spark local:
      * bigdl.sh
      * java -Xmx20g -Dspark.master="local[*]" -cp bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar  com.intel.analytics.bigdl.example.textclassification.TextClassifier  --baseDir $BASE_DIR  --batchSize 128  --env spark

    * Spark cluster:
      * bigdl.sh
      * spark-submit --master  spark://Gondolin-Node-040:7077 --driver-memory 10g --executor-memory 20g --class com.intel.analytics.bigdl.example.textclassification.TextClassifier  bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar  --coreNum 8 --nodeNum 4 --partitionNum 4 --batchSize 32  --baseDir ./postClassification/ --env spark
      * NB:
        * The total batch is: 32 * 4 as we specify nodeNum to be 4
        * partitionNum should be equal to nodeNum(We might relax this in the following release)
        * coreNum should be divided exactly by nodeNum

4. Verify:
   * Search accuracy from log:
   ``` 
   [Epoch 1 0/15964][Iteration 1][Wall Clock 0.0s] Train 128 in 1.669692383sec
   onds. Throughput is 76.66082764899336 records/second. Loss is 3.0336612164974213
   
   top1 accuracy is Accuracy(correct: 14749, count: 15964, accuracy: 0.9238912
      553244801)
