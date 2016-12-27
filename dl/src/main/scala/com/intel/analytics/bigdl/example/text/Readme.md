This simple text classification can achieve around 90% accuracy after 2 epoch training.
(It was first described in: https://github.com/fchollet/keras)

Embedding: 100-dimensional pre-trained GloVe embeddings of 400k words which trained on a 2014 dump of English Wikipedia.
Training data: "20 Newsgroup dataset" whichcontaining 20 categories and with totally 19997 texts.

Steps to run this example:
1)Download the Pre-train GloVe word embeddings from: http://nlp.stanford.edu/projects/glove/
2)Download the "20 Newsgroup dataset" as the training data from: http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
3)java -Xmx10g -cp bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-all-in-one.jar  com.intel.analytics.bigdl.example.text.classification.TextClassification  --baseDir $BASE_DIR  --batchSize 128
(NB: BASE_DIR is the root directory which containing the embedding and the training data. There are other optional parameters you can use as well. i.e: batchSize, trainingSplit)