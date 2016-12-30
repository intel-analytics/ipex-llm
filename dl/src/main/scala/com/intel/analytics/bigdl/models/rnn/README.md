#Recurrent Neural Network

Model that supports sequence to sequence processing

This is a implementation of Simple Recurrent Neural Network for Language Model. Please refer to the [original paper](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf) by Tomas Mikolov.

The implementation of RNN in this code is referred to [Keras Recurrent](https://keras.io/layers/recurrent/) documents.


##Get the BigDL files

Please build BigDL referring to [Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page).


##Prepare the Input Data
You can download one tiny Shakespeare Texts from [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

After downloading the text, please put it into one directory (e.g /opt/text/input.txt). The program will load in the original text file from this directory later.

###Sample Text

The input texts may look as follows:

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
./dist/bin/bigdl.sh -- java -cp bigdl_folder/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.rnn.Train -f /opt/text --core 4 --nEpochs 30 --learningRate 0.1

```

##Preprocessing

The <code>WordTokenizer.scala</code> in <code>rnn/Utils.scala</code> file is the source code to do the preprocessing procedure for the text.
It will create a dictionary with a key-value map format. The key is the word; the value is its corresponding index.
All the words in the dictionary are selected with regards to their frequencies in the texts (top k frequency).
The dictionaryLength is passed to the class as a user-defined parameter. The script will add a SENTENCE_START and SENTENCE_END token to the begginning and end of the sentence.
A <code>mapped_data.txt</code> will be created to store the preprocessed texts. Each word is indicated by its index number in the dictionary.
Note that all the words that are not included in the dictionary will be indicated as an UNKNOWN_TOKEN index.
Both files will be saved to the <code>saveDirectory</code>, which is defined by user.

###Sample Sequence of Processed Data
```
      3998,3875,3690,3999
      3998,3171,3958,2390,3832,3202,3855,3983,3883,3999
      3998,3667,3999
      3998,3151,3883,3999
      3998,3875,3690,3999
```

##Data Loading
The <code>readSentence</code> function in <code>rnn/Utils.scala</code> file will load in the training data from disk. It will shuffle the input data and split them into training and testing parts by the ratio of 8:2.
The <code>Dataset.array()</code> is a pipeline that will load the data and transform them to the expected training format.

##Model
A SimpleRNN model is implemented in the <code>Model.scala</code> script. It is a one hidden layer recurrent neural network with arbitrary hidden circles.
User can define the inputSize, hiddenSize, outputSize and bptt (back propagation through time) parameter to fine-tune the model.

##Expected Output
Users can see the Loss of the model printed by the program. The Loss means the perplexity of the model. The lower, the better.
```
[Epoch 1 1/26220][Iteration 1][Wall Clock 13.925086008s] loss is 8.333075046539307, iteration time is 0.04556048s data fetch time is 8.10813E-4s, train time 0.044749667s.
```

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
  --coreNumber   [engine core numbers, e.g 4 for a desktop]
```
