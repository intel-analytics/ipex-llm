#
# Copyright 2016 The BigDL Authors.
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

import nlp_architect.models.intent_extraction as intent_models
from bigdl.orca.tfpark.text.keras.text_model import TextKerasModel


class IntentEntity(TextKerasModel):
    """
    A multi-task model used for joint intent extraction and slot filling.

    This model has two inputs:
    - word indices of shape (batch, sequence_length)
    - character indices of shape (batch, sequence_length, word_length)
    This model has two outputs:
    - intent labels of shape (batch, num_intents)
    - entity tags of shape (batch, sequence_length, num_entities)

    :param num_intents: Positive int. The number of intent classes to be classified.
    :param num_entities: Positive int. The number of slot labels to be classified.
    :param word_vocab_size: Positive int. The size of the word dictionary.
    :param char_vocab_size: Positive int. The size of the character dictionary.
    :param word_length: Positive int. The max word length in characters. Default is 12.
    :param word_emb_dim: Positive int. The dimension of word embeddings. Default is 100.
    :param char_emb_dim: Positive int. The dimension of character embeddings. Default is 30.
    :param char_lstm_dim: Positive int. The hidden size of character feature Bi-LSTM layer.
                          Default is 30.
    :param tagger_lstm_dim: Positive int. The hidden size of tagger Bi-LSTM layers. Default is 100.
    :param dropout: Dropout rate. Default is 0.2.
    :param optimizer: Optimizer to train the model.
                      If not specified, it will by default to be tf.train.AdamOptimizer().
    """
    def __init__(self, num_intents, num_entities, word_vocab_size,
                 char_vocab_size, word_length=12, word_emb_dim=100, char_emb_dim=30,
                 char_lstm_dim=30, tagger_lstm_dim=100, dropout=0.2, optimizer=None):
        super(IntentEntity, self).__init__(intent_models.MultiTaskIntentModel(use_cudnn=False),
                                           optimizer,
                                           word_length=word_length,
                                           num_labels=num_entities,
                                           num_intent_labels=num_intents,
                                           word_vocab_size=word_vocab_size,
                                           char_vocab_size=char_vocab_size,
                                           word_emb_dims=word_emb_dim,
                                           char_emb_dims=char_emb_dim,
                                           char_lstm_dims=char_lstm_dim,
                                           tagger_lstm_dims=tagger_lstm_dim,
                                           dropout=dropout)

    @staticmethod
    def load_model(path):
        """
        Load an existing IntentEntity model (with weights) from HDF5 file.

        :param path: String. The path to the pre-defined model.
        :return: IntentEntity.
        """
        labor = intent_models.MultiTaskIntentModel(use_cudnn=False)
        model = TextKerasModel._load_model(labor, path)
        model.__class__ = IntentEntity
        return model
