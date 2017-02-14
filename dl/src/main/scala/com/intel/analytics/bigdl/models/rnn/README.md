#Recurrent Neural Network

Model that supports sequence to sequence processing

This is an implementation of Simple Recurrent Neural Networks for Language Modeling. Please refer to the [original paper](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf) by Tomas Mikolov.

The implementation of RNNs in this code is referred to in the [Keras Recurrent](https://keras.io/layers/recurrent/) documentation.


##Get the BigDL files

Please build BigDL referring to [Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page).


##Prepare the Input Data
You can download the Tiny Shakespeare Texts corpus from [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

After downloading the text, please place it into an appropriate directory (e.g /opt/text/input.txt). The program will later read in the original text file from this directory.

###Sample Text

The input text may look as follows:

```
      First Citizen:
      Before we proceed any further, hear me speak.

      All:
      Speak, speak.

      First Citizen:
      You are all resolved rather to die than to famish?
```

##Train the Model
Example command:
```bash
./dist/bin/bigdl.sh -- java -cp bigdl_folder/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.rnn.Train -f /opt/text -s /opt/save -n 1 -c 4 -b 12 --sent /opt/sent.bin --token token.bin --env local

```

##Test the Model

Please create a <code>test.txt</code> file under the folder in which you save your dictionary during training process.
A sample <code>test.txt</code> can be as follows. Each line starts with several trigger words and ends with a period. The test script will load in the trained model and <code>test.txt</code>, then it will generate the following words per line.
```
Long live the.
Upon her head.
Her hair, nor loose.
A thousand favours.
This said, in top of rage.
When forty winters shall.
And dig deep trenches in.
Then being ask'd where.
Look in thy glass,.
Now is the time that.
Thou dost beguile.
But if thou live,.
Each eye that saw him.
```

Example command:
```bash
./dist/bin/bigdl.sh -- java -cp bigdl_folder/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.rnn.Test -f /textdirectory --model /modeldirectory/model.iterationNumber --state /modeldirectory/state.iterationNumber -c 4 --words 20
```

##Preprocessing

The <code>SentenceSplitter</code>, <code>SentenceTokenizer</code> classes use [Apache OpenNLP library](https://opennlp.apache.org/).
The trained model <code>en-token.bin</code> and <code>en-sent.bin</code> can be reached via [here](http://opennlp.sourceforge.net/models-1.5/).
The <code>Dictionary.scala</code> accepts an array of string indicating for tokenized sentences or a file directory storing all the vocabulary.
It provides profuse API to reach the contents of dictionary. Such as <code>vocabSize()</code>, <code>word2Index()</code>, <code>vocabulary()</code>.
The dictionary information will be saved to <code>/opt/save/dictionary.txt</code>.

###Sample Sequence of Processed Data
```
      3998,3875,3690,3999
      3998,3171,3958,2390,3832,3202,3855,3983,3883,3999
      3998,3667,3999
      3998,3151,3883,3999
      3998,3875,3690,3999
```

##Model
A SimpleRNN model is implemented in the <code>Model.scala</code> script. It is a one hidden layer recurrent neural network with arbitrary hidden circles.
Users can define the <code>inputSize</code>, <code>hiddenSize</code>, <code>outputSize</code> and <code>bptt</code> (back propagation through time) parameters to fine-tune the model.

##Expected Training Output
Users can see the Loss of the model printed by the program. The Loss, in this case, is the perplexity of the language model. The lower, the better.
```
INFO  LocalOptimizer$:152 - [Epoch 1 12/7484][Iteration 1][Wall Clock 1.872321596s] loss is 8.302657127380371, iteration time is 1.872321596s data fetch time is 0.018143926s, train time 1.85417767s. Throughput is 6.409155363927127 record / second
INFO  LocalOptimizer$:152 - [Epoch 1 24/7484][Iteration 2][Wall Clock 3.105456523s] loss is 8.141095399856567, iteration time is 1.233134927s data fetch time is 0.006831927s, train time 1.226303s. Throughput is 9.731295203189067 record / second
INFO  LocalOptimizer$:152 - [Epoch 1 36/7484][Iteration 3][Wall Clock 4.293285676s] loss is 7.992086887359619, iteration time is 1.187829153s data fetch time is 0.007491824s, train time 1.180337329s. Throughput is 10.102462942328541 record / second
INFO  LocalOptimizer$:152 - [Epoch 1 48/7484][Iteration 4][Wall Clock 5.38083842s] loss is 7.7980124950408936, iteration time is 1.087552744s data fetch time is 0.007037109s, train time 1.080515635s. Throughput is 11.033947609625082 record / second
```

##Expected Testing Output
The test program will load the dictionary and test.txt(with several trigger words as the start tokens of the sentences) and generate the predicted output. The number of words to predict is defined by user with arguments --words.
