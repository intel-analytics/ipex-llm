There are a number of built-in __compiled__ text models in Analytics Zoo TFPark for Natural Language Processing (NLP) tasks based on [KerasModel](../TFPark/model/).

After constructing a text model, you can directly call [fit](../TFPark/model/#fit), [evaluate](../TFPark/model/#evaluate) or [predict](../TFPark/model/#predict) 
in a distributed fashion. 
See [here](../../ProgrammingGuide/text-models/) for more instructions.

__Remarks__:

- You need to install __tensorflow==1.15.0__ on your driver node.
- Your operating system (OS) is required to be one of the following 64-bit systems:
__Ubuntu 16.04 or later__ and __macOS 10.12.6 or later__.
- To run on other systems, you need to manually compile the TensorFlow source code. Instructions can
  be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).


---
## **Intent Extraction**
This is a multi-task model used for joint intent extraction and slot filling.

This model has two inputs:

- word indices of shape (batch, sequence_length)
- character indices of shape (batch, sequence_length, word_length)

This model has two outputs:

- intent labels of shape (batch, num_intents)
- entity tags of shape (batch, sequence_length, num_entities)

```python
from zoo.tfpark.text.keras import IntentEntity

model = IntentEntity(num_intents, num_entities, word_vocab_size, char_vocab_size, word_length=12, word_emb_dim=100, char_emb_dim=30, char_lstm_dim=30, tagger_lstm_dim=100, dropout=0.2, optimizer=None)
```

* `num_intents`: Positive int. The number of intent classes to be classified.
* `num_entities`: Positive int. The number of slot labels to be classified.
* `word_vocab_size`: Positive int. The size of the word dictionary.
* `char_vocab_size`: Positive int. The size of the character dictionary.
* `word_length`: Positive int. The max word length in characters. Default is 12.
* `word_emb_dim`: Positive int. The dimension of word embeddings. Default is 100.
* `char_emb_dim`: Positive int. The dimension of character embeddings. Default is 30.
* `char_lstm_dim`: Positive int. The hidden size of character feature Bi-LSTM layer. Default is 30.
* `tagger_lstm_dim`: Positive int. The hidden size of tagger Bi-LSTM layers. Default is 100.
* `dropout`: Dropout rate. Default is 0.2.
* `optimizer`: Optimizer to train the model. If not specified, it will by default to be tf.train.AdamOptimizer().


**Model Save and Load**

Save the `IntentEntity` model to a single HDF5 file.

```python
model.save_model(path)
```

Load an existing `IntentEntity` model (with weights) from HDF5 file.

```python
from zoo.tfpark.text.keras import IntentEntity

model = IntentEntity.load_model(path)
```


---
## **Named Entity Recognition**
This model is used for named entity recognition using Bidirectional LSTM with
Conditional Random Field (CRF) sequence classifier.

This model has two inputs:

- word indices of shape (batch, sequence_length)
- character indices of shape (batch, sequence_length, word_length)

This model outputs entity tags of shape (batch, sequence_length, num_entities).

```python
from zoo.tfpark.text.keras import NER

model = NER(num_entities, word_vocab_size, char_vocab_size, word_length=12, word_emb_dim=100, char_emb_dim=30, tagger_lstm_dim=100, dropout=0.5, crf_mode='reg', optimizer=None)
```

* `num_entities`: Positive int. The number of entity labels to be classified.
* `word_vocab_size`: Positive int. The size of the word dictionary.
* `char_vocab_size`: Positive int. The size of the character dictionary.
* `word_length`: Positive int. The max word length in characters. Default is 12.
* `word_emb_dim`: Positive int. The dimension of word embeddings. Default is 100.
* `char_emb_dim`: Positive int. The dimension of character embeddings. Default is 30.
* `tagger_lstm_dim`: Positive int. The hidden size of tagger Bi-LSTM layers. Default is 100.
* `dropout`: Dropout rate. Default is 0.5.
* `crf_mode`: String. CRF operation mode. Either 'reg' or 'pad'. Default is 'reg'. 
                     'reg' for regular full sequence learning (all sequences have equal length). 
                     'pad' for supplied sequence lengths (useful for padded sequences). 
                     For 'pad' mode, a third input for sequence_length (batch, 1) is needed.
* `optimizer`: Optimizer to train the model. If not specified, it will by default to be tf.keras.optimizers.Adam(0.001, clipnorm=5.).


**Model Save and Load**

Save the `NER` model to a single HDF5 file.

```python
model.save_model(path)
```

Load an existing `NER` model (with weights) from HDF5 file.

```python
from zoo.tfpark.text.keras import NER

model = NER.load_model(path)
```


---
## **POS Tagging**
This model is used as Part-Of-Speech(POS)-tagger and chunker for sentence tagging, which contains three
Bidirectional LSTM layers.

This model can have one or two input(s):

- word indices of shape (batch, sequence_length)
- character indices of shape (batch, sequence_length, word_length) (if char_vocab_size is not None)

This model has two outputs:

- pos tags of shape (batch, sequence_length, num_pos_labels)
- chunk tags of shape (batch, sequence_length, num_chunk_labels)

```python
from zoo.tfpark.text.keras import SequenceTagger

model = NER(num_pos_labels, num_chunk_labels, word_vocab_size, char_vocab_size=None, word_length=12, feature_size=100, dropout=0.2, classifier='softmax', optimizer=None)
```

* `num_pos_labels`: Positive int. The number of pos labels to be classified.
* `num_chunk_labels`: Positive int. The number of chunk labels to be classified.
* `word_vocab_size`: Positive int. The size of the word dictionary.
* `char_vocab_size`: Positive int. The size of the character dictionary.
Default is None and in this case only one input, namely word indices is expected.
* `word_length`: Positive int. The max word length in characters. Default is 12.
* `feature_size`: Positive int. The size of Embedding and Bi-LSTM layers. Default is 100.
* `dropout`: Dropout rate. Default is 0.5.
* `classifier`: String. The classification layer used for tagging chunks. 
Either 'softmax' or 'crf' (Conditional Random Field). Default is 'softmax'.
* `optimizer`: Optimizer to train the model. If not specified, it will by default to be tf.train.AdamOptimizer().


**Model Save and Load**

Save the `SequenceTagger` model to a single HDF5 file.

```python
model.save_model(path)
```

Load an existing `SequenceTagger` model (with weights) from HDF5 file.

```python
from zoo.tfpark.text.keras import SequenceTagger

model = SequenceTagger.load_model(path)
```
