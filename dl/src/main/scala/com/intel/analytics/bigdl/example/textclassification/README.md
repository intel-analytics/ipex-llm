## Summary
 This example use a (pre-trained GloVe embedding) to convert word to vector,
 and uses it to train the text classification model on a 20 Newsgroup dataset
 with 20 different categories. This model can achieve around 90% accuracy after 2 epochs training.
(It was first described in: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
## Data
* Embedding: 100-dimensional pre-trained GloVe embeddings of 400k words which trained on a 2014 dump of English Wikipedia.
* Training data: "20 Newsgroup dataset" which containing 20 categories and with totally 19997 texts.

## Steps to run this example:
1. Download [Pre-train GloVe word embeddings](http://nlp.stanford.edu/data/glove.6B.zip)
2. Download [20 Newsgroup dataset](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html) as the training data
3. Run the commands:

    * Spark local:
      * Execute: bigdl.sh
      * spark-submit --master "local[*]" --driver-memory 20g --class com.intel.analytics.bigdl.example.textclassification.TextClassifier  bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --batchSize 128  --baseDir $BASE_DIR

    * Spark cluster:
      * Execute: bigdl.sh
      * spark-submit --master  $MASTER --driver-memory 10g --executor-memory 20g --class com.intel.analytics.bigdl.example.textclassification.TextClassifier  bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar  --coreNum 8 --nodeNum 4 --batchSize 32  --baseDir $BASE_DIR
      * NOTE: The total batch is: 32 * 4 as we specify nodeNum to be 4

4. Verify:
   * Search accuracy from log:
   ``` 
   [Epoch 1 0/15964][Iteration 1][Wall Clock 0.0s] Train 128 in 1.669692383sec
   onds. Throughput is 76.66082764899336 records/second. Loss is 3.0336612164974213
   
   top1 accuracy is Accuracy(correct: 14749, count: 15964, accuracy: 0.9238912
      553244801)
