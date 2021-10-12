Analytics Zoo provides a built-in BERTClassifier in TFPark for Natural Language Processing (NLP) classification tasks based on [TFEstimator](../TFPark/estimator/) and BERT.

Bidirectional Encoder Representations from Transformers (BERT) is Google's state-of-the-art pre-trained NLP model.
You may refer to [here](https://github.com/google-research/bert) for more details.

BERTClassifier is a pre-built TFEstimator that takes the hidden state of the first token to do classification.

__Remarks__:

- You need to install __tensorflow==1.15.0__ on your driver node.
- Your operating system (OS) is required to be one of the following 64-bit systems:
__Ubuntu 16.04 or later__ and __macOS 10.12.6 or later__.
- To run on other systems, you need to manually compile the TensorFlow source code. Instructions can
  be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).

After constructing a BERTClassifier, you can directly call [train](../TFPark/estimator/#train), [evaluate](../TFPark/estimator/#evaluate) or [predict](../TFPark/estimator/#predict) 
in a distributed fashion. 
See [here](../../ProgrammingGuide/bert-classifier/) for more instructions.

```python
from zoo.tfpark.text.estimator import BERTClassifier

estimator = BERTClassifier(num_classes, bert_config_file, init_checkpoint=None, use_one_hot_embeddings=False, optimizer=None, model_dir=None)
```

* `num_classes`: Positive int. The number of classes to be classified.
* `bert_config_file`: The path to the json file for BERT configurations.
* `init_checkpoint`: The path to the initial checkpoint of the pre-trained BERT model if any. Default is None.
* `use_one_hot_embeddings`: Boolean. Whether to use one-hot for word embeddings. Default is False.
* `optimizer`: The optimizer used to train the estimator. It can either be an instance of 
tf.train.Optimizer or the corresponding string representation. Default is None if no training is involved.
* `model_dir`: The output directory for model checkpoints to be written if any. Default is None.
