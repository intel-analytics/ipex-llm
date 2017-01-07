#Recurrent Neural Network

Model that supports sequence to sequence processing

This is an implementation of Simple Recurrent Neural Networks for Language Modeling. Please refer to the [original paper](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf) by Tomas Mikolov.

The implementation of RNNs in this code is referred to in the [Keras Recurrent](https://keras.io/layers/recurrent/) documentation.


##Get the BigDL files

Please build BigDL referring to [Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page).


##Prepare the Input Data
You can download the Tiny Shakespeare Texts corpus from [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

After downloading the text, please put it into an appropriate directory (e.g /opt/data/).

The program will load in the text files in this directory for training and validation. Then, it will save the generated dictionary.txt, dicard.txt and mapped_data.txt to the saveFolder.

Please use -f or --dataFolder to indicate the /opt/data directory.

Please use -s or --saveFolder to indicate the /opt/save directory.

Three files will be generated after processing the raw texts. They are:

* dictionary.txt
* discard.txt
* mapped_data.txt

The <code>dictionary.txt</code> saves the dictionary collected from raw texts. Note that the length of this dictionary is defined by argument --vocab.
If the vocabSize is larger than the maximum words that can be collected, the program will use that maximum words as a substitute.

If the <code>mapped_data.txt</code> has already been generated, the program will skip the preprocessing step.

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

###Sample Sequence of Processed Data
```
      3998,3875,3690,3999
      3998,3171,3958,2390,3832,3202,3855,3983,3883,3999
      3998,3667,3999
      3998,3151,3883,3999
      3998,3875,3690,3999
```

##Train the Model
Example command:
```bash
./scripts/bigdl.sh -- java -cp /folder/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.rnn.Train -f /dataFolder -s /saveFolder -c 4 --vocab 10000 -h 100 --learningRate 0.1 -e 50 --bptt 5 --checkpoint /modeldirectory
```

##Test the Model
Example command:
```bash
./dist/bin/bigdl.sh -- java -cp bigdl_folder/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.rnn.Test -f /textdirectory --test test.txt --model /modeldirectory/model.iterationNumber --state /modeldirectory/state.iterationNumber -c 4 --words 20
```

##Preprocessing

The <code>WordTokenizer</code> class in the <code>rnn/Utils.scala</code> file implements the preprocessing procedure for the input text.
The text will be tokenized by a [stanford tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml).
It will create a dictionary with a key-value map format. Each key is a word from the input text data; each value is a corresponding index of such a word.
The words in the dictionary are selected with regard to their frequencies in the texts (top-k frequencies).
The <code>dictionaryLength</code> is passed to the class as a user-defined parameter. The script will add  <code>SENTENCE_START</code> and <code>SENTENCE_END</code> tokens to the beginning and end of every sentence.
A <code>mapped_data.txt</code> file will be created to store the preprocessed texts. Each word is indicated by its index number in the dictionary.
Note that all the words not included in the dictionary will merely be indicated as an <code>UNKNOWN_TOKEN</code> index.
Both files will be saved to the <code>saveDirectory</code>, which is defined by the user.


##Data Loading
The <code>loadData</code> function in <code>rnn/Utils.scala</code> file will load in the training data from disk. It will shuffle the input data and split this data into training and testing parts with a ratio of 8:2.
The <code>Dataset.array()</code> is a pipeline that will load the data and transform it to the expected training format.

##Model
A SimpleRNN model is implemented in the <code>Model.scala</code> script. It is a one hidden layer recurrent neural network with arbitrary hidden circles.
Users can define the <code>inputSize</code>, <code>hiddenSize</code>, <code>outputSize</code> and <code>bptt</code> (back propagation through time) parameters to fine-tune the model.

##Training
The model will be trained sentence by sentence. For each epoch, the program will save a snapshot of the trained model to the directory defined by --checkpoint parameter.

##Testing
The basic idea of testing a language model is to let the model generate sentence automatically. Thus, a test.txt file will be provided by the user. In this file,
users have to give some trigger words (>= 2 words). Please use period "." token to indicate the end of the trigger words for each sentence.

For example, some sample trigger words can be:
```
Long live the.
For this relief.
A man he says.
Therefore, I have entreated.
What art thou that.
Welcome, Horatio.
I have seen.
...
```

Then the program will load these trigger words and generate the next serious of words automatically. The number of words to generate for each sentence is defined by --words parameter. The results will be saved under the --folder directory.

For example, the expected outputs for these trigger words by the model can be:
```
Long live the .  of the sky .   conversing .   the drained of the quietus .   the scaping of the furrows .   and Clogg of the
 For this relief .  , but what the cause .   the truth of the Cephalenian .   unhappily of the pointing of the stair .   the unload of the
 A man he says .   The outflies of the divan .   shreds of the breeds of the acclaim .   the truth of the leagues .   table-book of the
 Therefore , I have entreated .  ?   The errs of the cap-à-pie of the sky .   The enchased of the bodes of the cock .   The Known 's  ,
 What art thou that .  , and the King of the Weigh .   the truth of the Lycians .   heaven-illumined of the heaven-illumined of the freshness .   the truth
 Welcome , Horatio .   ' t ?  '' and your father 's the King .   unlamented of the cherished of the Successless .   the sentences of the Strew
 I have seen .   And let them to the rest .   rowers of the crest of the meditates .   the thicker of the unbelieving .   and conversing
```

##Expected Training Output
Users can see the Loss of the model printed by the program. The Loss, in this case, is the perplexity of the language model. The lower, the better.
```
INFO  LocalOptimizer$:152 - [Epoch 1 1/26221][Iteration 1][Wall Clock 0.225452714s] loss is 8.3017578125, iteration time is 0.225452714s data fetch time is 0.001966759s, train time 0.223485955s. Throughput is 4.435519902412885 record / second
```
