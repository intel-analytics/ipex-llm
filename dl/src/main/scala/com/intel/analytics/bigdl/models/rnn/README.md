#Recurrent Neural Network

Model that supports sequence to sequence processing

##Dataset

Shakespeare texts are used as a toy example to train a language model.

The input.txt is a pure text file format, which contains the Shakespeare's poems.

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

##Preprocessing

The <code>WordTokenizer.scala</code> in <code>rnn/Utils.scala</code> file is the source code to do the preprocessing procedure for the text.
It will create a dictionary with a key-value map format. The key is the word; the value is its corresponding index.
All the words in the dictionary are selected with regards to their frequencies in the texts (top k frequency).
The dictionaryLength is passed to the class as a user-defined parameter. The script will add a SENTENCE_START and SENTENCE_END token to the begginning and end of the sentence.
A <code>mapped_data.txt</code> will be created to store the preprocessed texts. Each word is indicated by its index number in the dictionary.
Note that all the words that are not included in the dictionary will be indicated as an UNKNOWN_TOKEN index.
Both files will be saved to the <code>saveDirectory</code>, which is defined by user.

###Sample Data
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

##Training
Note that we train the rnn sentence by sentence. Thus the following codes are necessary settings.
<br>
<code>
Engine.setCoreNumber(1)
</code><br>
<code>
Engine.model.setPoolSize(param.coreNumber)
</code><br>

The <code>optimizer</code> method is a highly optimized class. Users can simply pass their parameters to this class to initiate training.
For example, the validation method, number of epochs to train, learning rate, momentum, etc.

In our example, we set Loss as the validation method, and learning rate is set to be 0.1

##Parameters
Almost all the parameters that can tune the model are user defined.
Users can define their training epochs, learning rate, input file folder, etc as the command line arguments.
Note that user must define the coreNumber (for example: 4 for a desktop).

##Sample Command Line
<code>
java -cp bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar:spark-assembly-1.5.1-hadoop2.6.0.jar com.intel.analytics.bigdl.models.rnn.Train -f ./ --core 4 --nEpochs 30 --learningRate 0.1
</code>
