# Recurrent Neural Network

Model that supports sequence to sequence processing

This is an implementation of Simple Recurrent Neural Networks for Language Modeling. Please refer to the [original paper](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf) by Tomas Mikolov.

The implementation of RNNs in this code is referred to in the [Keras Recurrent](https://keras.io/layers/recurrent/) documentation.


## Get the BigDL files

Please build BigDL referring to [Build Page](https://bigdl-project.github.io/master/#ScalaUserGuide/install-build-src/).


## Prepare the Input Data
You can download the Tiny Shakespeare Texts corpus from [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

After downloading the text, please place it into an appropriate directory (e.g /opt/text/input.txt). Please separate it into train.txt and val.txt. In our case, we just select 80 percentage of the input to be train and remaining 20 percentage to be val. The program will later read in the original text file from this directory.
```shell
export LANG=en_US.UTF-8
head -n 8000 input.txt > val.txt
tail -n +8000 input.txt > train.txt
```

### Sample Text

The input text may look as follows:

```
      First Citizen:
      Before we proceed any further, hear me speak.

      All:
      Speak, speak.

      First Citizen:
      You are all resolved rather to die than to famish?
```

## Train the Model
Example command:
```bash
spark-submit \
--master spark://... \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--class com.intel.analytics.bigdl.models.rnn.Train \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f /path/inputdata/ -s /path/saveDict --checkpoint /path/model/ --batchSize 12
```

## Test the Model
Please create a <code>test.txt</code> file under the folder <code> /path/saveDict</code> in which you save your dictionary during training process.
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
spark-submit \
--master spark://... \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.rnn.Test \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f /path/saveDict --model /path/model/model.iterationNumber --words 20
```

## Preprocessing

The <code>SentenceSplitter</code>, <code>SentenceTokenizer</code> classes use [Apache OpenNLP library](https://opennlp.apache.org/).
The trained model <code>en-token.bin</code> and <code>en-sent.bin</code> can be reached via [here](http://opennlp.sourceforge.net/models-1.5/).
Please upload these two files onto HDFS and pass the host and path to the command line arguments.

Example command:
```bash
spark-submit \
--master spark://... \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--class com.intel.analytics.bigdl.models.rnn.Train \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f /path/inputdata/ -s /path/saveDict --checkpoint /path/model/ --batchSize 12 \
--sent hdfs://127.0.0.1:9001/tokenizer/en-sent.bin --token hdfs://127.0.0.1:9001/tokenizer/en-token.bin
```

The <code>Dictionary.scala</code> accepts an array of string indicating for tokenized sentences or a file directory storing all the vocabulary.
It provides profuse API to reach the contents of dictionary. Such as <code>vocabSize()</code>, <code>word2Index()</code>, <code>vocabulary()</code>.
The dictionary information will be saved to <code>/opt/save/dictionary.txt</code>.

### Sample Sequence of Processed Data
```
      3998,3875,3690,3999
      3998,3171,3958,2390,3832,3202,3855,3983,3883,3999
      3998,3667,3999
      3998,3151,3883,3999
      3998,3875,3690,3999
```

## Model
A SimpleRNN model is implemented in the <code>Model.scala</code> script. It is a one hidden layer recurrent neural network with arbitrary hidden circles.
Users can define the <code>inputSize</code>, <code>hiddenSize</code>, <code>outputSize</code> and <code>bptt</code> (back propagation through time) parameters to fine-tune the model.

## Expected Training Output
Users can see the Loss of the model printed by the program. The Loss, in this case, is the perplexity of the language model. The lower, the better.
```
INFO  DistriOptimizer$:247 - [Epoch 1 0/6879][Iteration 1][Wall Clock 0.0s] Train 12 in 4.926679827seconds. Throughput is 2.4357176 records/second. Loss is 8.277311. Current learning rate is 0.1.
INFO  DistriOptimizer$:247 - [Epoch 1 12/6879][Iteration 2][Wall Clock 4.926679827s] Train 12 in 2.622718594seconds. Throughput is 4.575405 records/second. Loss is 8.07377. Current learning rate is 0.1.
INFO  DistriOptimizer$:247 - [Epoch 1 24/6879][Iteration 3][Wall Clock 7.549398421s] Train 12 in 2.478575083seconds. Throughput is 4.8414917 records/second. Loss is 7.8527904. Current learning rate is 0.1.
INFO  DistriOptimizer$:247 - [Epoch 1 36/6879][Iteration 4][Wall Clock 10.027973504s] Train 12 in 2.475138056seconds. Throughput is 4.8482146 records/second. Loss is 7.581617. Current learning rate is 0.1.
...
```
