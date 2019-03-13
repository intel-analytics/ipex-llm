#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from nlp_architect.models import chunker
from zoo.tfpark.text import TextKerasModel


class SequenceTagger(TextKerasModel):
    """
    The model used as POS-tagger and chunker for sentence tagging, which contains three
    Bidirectional LSTM layers.

    This model can have one or two input(s):
    - word indices of shape (batch, sequence_length)
    *If char_vocab_size is not None:
    - character indices of shape (batch, sequence_length, word_length)
    This model has two outputs:
    - pos tags of shape (batch, sequence_length, num_pos_labels)
    - chunk tags of shape (batch, sequence_length, num_chunk_labels)

    :param num_pos_labels: Positive int. The number of pos labels to be classified.
    :param num_chunk_labels: Positive int. The number of chunk labels to be classified.
    :param word_vocab_size: Positive int. The size of the word dictionary.
    :param char_vocab_size: Positive int. The size of the character dictionary.
                            Default is None and in this case only one input, namely word indices
                            is expected.
    :param word_length: Positive int. The max word length in characters. Default is 12.
    :param feature_size: Positive int. The size of Embedding and Bi-LSTM layers. Default is 100.
    :param dropout: Dropout rate. Default is 0.5.
    :param classifier: String. The classification layer used for tagging chunks.
                       Either 'softmax' or 'crf' (Conditional Random Field). Default is 'softmax'.
    :param optimizer: Optimizer to train the model. If not specified, it will by default
                      to be tf.train.AdamOptimizer().
    """
    def __init__(self, num_pos_labels, num_chunk_labels, word_vocab_size,
                 char_vocab_size=None, word_length=12, feature_size=100, dropout=0.2,
                 classifier='softmax', optimizer=None):
        classifier = classifier.lower()
        assert classifier in ['softmax', 'crf'], "classifier should be either softmax or crf"
        super(SequenceTagger, self).__init__(chunker.SequenceTagger(use_cudnn=False),
                                             vocabulary_size=word_vocab_size,
                                             num_pos_labels=num_pos_labels,
                                             num_chunk_labels=num_chunk_labels,
                                             char_vocab_size=char_vocab_size,
                                             max_word_len=word_length,
                                             feature_size=feature_size,
                                             dropout=dropout,
                                             classifier=classifier,
                                             optimizer=optimizer)

    @staticmethod
    def load_model(path):
        labor = chunker.SequenceTagger(use_cudnn=False)
        model = TextKerasModel._load_model(labor, path)
        model.__class__ = SequenceTagger
        return model
