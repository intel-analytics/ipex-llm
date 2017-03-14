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
./dist/bin/bigdl.sh -l -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.rnn.Train -f /opt/text --nEpochs 30 --learningRate 0.1
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
./dist/bin/bigdl.sh -l -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.rnn.Test -f /textdirectory --model /modeldirectory/model.iterationNumber --words 20
```

##Preprocessing

The <code>WordTokenizer</code> class in the <code>rnn/Utils.scala</code> file implements the preprocessing procedure for the input text.
It will create a dictionary with a key-value map format. Each key is a word from the input text data; each value is a corresponding index of such a word.
The words in the dictionary are selected with regard to their frequencies in the texts (top-k frequencies).
The <code>dictionaryLength</code> is passed to the class as a user-defined parameter. The script will add  <code>SENTENCE_START</code> and <code>SENTENCE_END</code> tokens to the beginning and end of every sentence.
A <code>mapped_data.txt</code> file will be created to store the preprocessed texts. Each word is indicated by its index number in the dictionary.
Note that all the words not included in the dictionary will merely be indicated as an <code>UNKNOWN_TOKEN</code> index.
Both files will be saved to the <code>saveDirectory</code>, which is defined by the user.

###Sample Sequence of Processed Data
```
      3998,3875,3690,3999
      3998,3171,3958,2390,3832,3202,3855,3983,3883,3999
      3998,3667,3999
      3998,3151,3883,3999
      3998,3875,3690,3999
```

##Data Loading
The <code>readSentence</code> function in <code>rnn/Utils.scala</code> file will load in the training data from disk. It will shuffle the input data and split this data into training and testing parts with a ratio of 8:2.
The <code>Dataset.array()</code> is a pipeline that will load the data and transform it to the expected training format.

##Model
A SimpleRNN model is implemented in the <code>Model.scala</code> script. It is a one hidden layer recurrent neural network with arbitrary hidden circles.
Users can define the <code>inputSize</code>, <code>hiddenSize</code>, <code>outputSize</code> and <code>bptt</code> (back propagation through time) parameters to fine-tune the model.

##Expected Training Output
Users can see the Loss of the model printed by the program. The Loss, in this case, is the perplexity of the language model. The lower, the better.
```
INFO  LocalOptimizer$:152 - [Epoch 1 1/26221][Iteration 1][Wall Clock 0.225452714s] loss is 8.3017578125, iteration time is 0.225452714s data fetch time is 0.001966759s, train time 0.223485955s. Throughput is 4.435519902412885 record / second
```

##Expected Testing Output
The test program will load the dictionary and test.txt(with several trigger words as the start tokens of the sentences) and generate the predicted output. The number of words to predict is defined by user with arguments --words.


##Parameters
```
  --folder | -f  [the directory to reach the data and save generated dictionary]
  --learningRate [default 0.1]
  --momentum     [default 0.0]
  --weightDecay  [default 0.0]
  --dampening    [default 0.0]
  --hiddenSize   [the size of recurrent layer, default 40]
  --vocabSize    [the vocabulary size that users would like to set, default 4000]
  --bptt         [back propagation through time, default 4]
  --nEpochs      [number of epochs to train, default 30]
```
