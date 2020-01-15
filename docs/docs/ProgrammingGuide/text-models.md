There are a number of built-in __compiled__ text models in Analytics Zoo TFPark for Natural Language Processing (NLP) tasks based on [KerasModel](../APIGuide/TFPark/model/).

See [this page](../APIGuide/TFPark/text-models/) for more details about how to construct built-in models for intent extraction, named entity extraction and pos tagging. etc.

In this page, we show the general steps how to train and evaluate an [NER](../APIGuide/TFPark/text-models/#named-entity-recognition) model in a distributed fashion and use this model for distributed inference.
For other models, the steps are more or less quite similar.


__Remarks__:

- You need to install __tensorflow==1.15.0__ on your driver node.
- Your operating system (OS) is required to be one of the following 64-bit systems:
__Ubuntu 16.04 or later__, __macOS 10.12.6 or later__ and __Windows 7 or later__.
- To run on other systems, you need to manually compile the TensorFlow source code. Instructions can
  be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).


---
## **Model Construction**
You can easily construct a model for named entity recognition using the following API.

```python
from zoo.tfpark.text.keras import NER

model = NER(num_entities, word_vocab_size, char_vocab_size, word_length)
```


---
## **Data Preparation**
The NER model has two inputs: word indices and character indices.

Thus, each raw text record needs to go through word-wise tokenization, character-wise segmentation and alignment to the same target length for preprocessing.

If you are using numpy arrays, then the input `x` should be a list of two numpy arrays:

- `x_words` of shape (batch, sequence_length) for word indices
- `x_chars` of shape (batch, sequence_length, word_length) for character indices.
- `x = [x_words, x_char]`

If there are labels (for training and evaluation), `y` should be another numpy array of shape (batch, sequence_length, word_length) for entity tags.

Alternatively, you can construct a [TFDataSet](../ProgrammingGuide/tensorflow/#tfdataset) directly if you are dealing with RDD.
Each record in TFDataSet should contain word indices, character indices and labels (if any) as well.


---
## **Model Training**
You can easily call [fit](../APIGuide/TFPark/model/#fit) to train the NER model in a distributed fashion. You don't need to specify `y` if `x` is already a TFDataSet.

```python
model.fit(x, y, batch_size, epochs, distributed=True)
```


---
## **Model Evaluation**
You can easily call [evaluate](../APIGuide/TFPark/model/#evaluate) to evaluate the NER model in a distributed fashion. You don't need to specify `y` if `x` is already a TFDataSet.

```python
result = model.evaluate(x, y, distributed=True)
```


---
## **Model Save and Load**
After training, you can save the `NER` model to a single HDF5 file.

```python
model.save_model(path)
```

For inference, you can load a directly trained `NER` model (with weights) from HDF5 file.

```python
from zoo.tfpark.text.keras import NER

model = NER.load_model(path)
```


---
## **Model Inference**
You can easily call [predict](../APIGuide/TFPark/model/#predict) to use the trained NER model for distributed inference. Note that you don't necessarily need labels for prediction.

```python
predictions = model.predict(x, distributed=True)
```
