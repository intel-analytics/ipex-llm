# Language Model

This example refers to [tensorflow ptb example](https://www.tensorflow.org/tutorials/recurrent#language_modeling), which shows how to train a recurrent neural network on a challenging task of language modeling.

The core of our model consists of LSTM cells that process one word at a time and computes probabilities of the possible values for the next word in the sentence.

Here we use [Penn Tree Bank (PTB)](https://catalog.ldc.upenn.edu/ldc99t42) as training dataset, which is a popular benchmark for measuring the quality of these models, whilst being small and relatively fast to train.

## Get BigDL jar

Please build BigDL referring to [Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page).

## Prepare PTB Data
Download PTB dataset from [Tomas Mikolov's webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)
Then, extract the PTB dataset underneath your home directory as follows:

```tar xvfz simple-examples.tgz -C $HOME```

Dataset here we need is $HOME/simple-examples/data

## Train the Model
Example command
```
spark-submit \
--master spark://... \
--driver-memory 40g  \
--executor-memory 100g  \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--class com.intel.analytics.bigdl.example.languagemodel.PTBWordLM \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f $HOME/simple-examples/data -b 40 --checkpoint $HOME/model --numLayers 2 --vocab 10001 --hidden 650 --numSteps 35 --learningRate 0.005 -e 20 --learningRateDecay 0.001 --keepProb 0.5 --overWrite
```

In the above commands:
```-f```: where you put your PTB data
```-b```: Total batch size. It is expected that batch size is a multiple of core_number
```--checkpoint```: Where you cache the model/train_state snapshot.
```--learningRate```: learning rate for adagrad
```--learningRateDecay```: learning rate decay for adagrad
```--hidden```: hiddensize for lstm
```--vocabSize```: vocabulary size, default 10000
```--nEpochs```: epochs to run
```--numLayers```: numbers of lstm cell, default 2 lstm cells
```--numSteps```: number of words per record in LM
```--overWrite```: do overwrite when saving checkpoint
```--keepProb```: the probability to do dropout