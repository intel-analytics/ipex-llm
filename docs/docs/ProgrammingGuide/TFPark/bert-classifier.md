Analytics Zoo provides a built-in BERTClassifier in TFPark for Natural Language Processing (NLP) classification tasks based on [TFEstimator](../APIGuide/TFPark/estimator/) and BERT.

Bidirectional Encoder Representations from Transformers (BERT) is Google's state-of-the-art pre-trained NLP model.
You may refer to [here](https://github.com/google-research/bert) for more details.

BERTClassifier is a pre-built TFEstimator that takes the hidden state of the first token to do classification.

In this page, we show the general steps how to train and evaluate an [BERTClassifier](../APIGuide/TFPark/bert-classifier/) in a distributed fashion and use this estimator for distributed inference.

__Remarks__:

- You need to install __tensorflow==1.15.0__ on your driver node.
- Your operating system (OS) is required to be one of the following 64-bit systems:
__Ubuntu 16.04 or later__, __macOS 10.12.6 or later__ and __Windows 7 or later__.
- To run on other systems, you need to manually compile the TensorFlow source code. Instructions can
  be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).


---
## **BERTClassifier Construction**
You can easily construct an estimator for classification based on BERT using the following API.

```python
from zoo.tfpark.text.estimator import BERTClassifier

estimator = BERTClassifier(num_classes, bert_config_file, init_checkpoint, optimizer=tf.train.AdamOptimizer(learning_rate), model_dir="/tmp/bert")
```


---
## **Data Preparation**
BERT has three inputs of the same sequence length: input_ids, input_mask and token_type_ids. 

The preprocessing steps should follow BERT's conventions. You may refer to BERT TensorFlow [run_classifier example](https://github.com/google-research/bert/blob/master/run_classifier.py) for more details.

To construct the input function for BERTClassifier, you can use the following API:
```python
from zoo.tfpark.text.estimator import bert_input_fn

input_fn = bert_input_fn(rdd, max_seq_length, batch_size)
```

- For training and evaluation, each element in rdd should be a tuple: (feature dict, label). Label is supposed to be an integer.
- For prediction, each element in rdd should be a feature dict.
- The keys of feature dict should be `input_ids`, `input_mask` and `token_type_ids` and the values should be the corresponding preprocessed results of max_seq_length for a record.


---
## **Estimator Training**
You can easily call [train](../APIGuide/TFPark/estimator/#train) to train the BERTClassifier in a distributed fashion.

```python
estimator.train(train_input_fn, steps)
```

You can find the trained checkpoints saved under `model_dir`, which is specified when you initiate BERTClassifier.


---
## **Estimator Evaluation**
You can easily call [evaluate](../APIGuide/TFPark/estimator/#evaluate) to evaluate the BERTClassifier in a distributed fashion using top1 accuracy.

```python
result = estimator.evaluate(eval_input_fn, eval_methods=["acc"])
```


---
## **Estimator Inference**
You can easily call [predict](../APIGuide/TFPark/estimator/#predict) to use the trained BERTClassifier for distributed inference.

```python
predictions_rdd = estimator.predict(test_input_fn)
```
