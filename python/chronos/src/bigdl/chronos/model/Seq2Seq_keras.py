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

import tensorflow as tf
from keras import backend as K
from bigdl.nano.tf.keras import Model
from keras.layers import LSTM, Dense, Lambda


class LSTMSeq2Seq(Model):
    def __init__(self,
                 input_feature_num,
                 future_seq_len,
                 output_feature_num,
                 lstm_hidden_dim,
                 lstm_layer_num,
                 dropout,
                 teacher_forcing):
        super(LSTMSeq2Seq, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layer_num = lstm_layer_num
        self.future_seq_len = future_seq_len
        self.output_feature_num = output_feature_num
        self.input_feature_num = input_feature_num
        self.dropout = dropout
        self.teacher_forcing = teacher_forcing
        self.encoder, self.decoder = [], []
        for i in range(self.lstm_layer_num):
            self.encoder.append(LSTM(self.lstm_hidden_dim,
                                     return_sequences=True,
                                     return_state=True,
                                     dropout=self.dropout,
                                     name="encoder_lstm_"+str(i)))
            self.decoder.append(LSTM(self.lstm_hidden_dim,
                                     return_sequences=True,
                                     return_state=True,
                                     dropout=self.dropout,
                                     name="decoder_lstm_"+str(i)))
        self.fc = Dense(self.output_feature_num, activation="linear")

    def call(self, input_seq, target_seq=None):
        states_value = None
        for encoder_layers in self.encoder:
            _, *states_value = encoder_layers(input_seq,
                                              initial_state=states_value)
        decoder_inputs = input_seq[:, -1, :self.output_feature_num]
        decoder_inputs = K.reshape(decoder_inputs,
                                   shape=(len(input_seq), 1, decoder_inputs.shape[-1]))
        all_outputs = []
        for i in range(self.future_seq_len):
            for decoder_layers in self.decoder:
                decoder_outputs, *states_value = decoder_layers(decoder_inputs,
                                                                initial_state=states_value)
                decoder_inputs = decoder_outputs
            decoder_outputs = self.fc(decoder_outputs)
            # Update states
            all_outputs.append(decoder_outputs)
            if not self.teacher_forcing or target_seq is None:
                # no teaching force
                decoder_inputs = decoder_outputs
            else:
                # with teaching force
                decoder_inputs = target_seq[:, i:i+1, :]
        decoded_seq = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
        return decoded_seq


def model_creator(config):
    model = LSTMSeq2Seq(input_feature_num=config["input_feature_num"],
                        output_feature_num=config["output_feature_num"],
                        future_seq_len=config["future_seq_len"],
                        lstm_hidden_dim=config.get("lstm_hidden_dim", 128),
                        lstm_layer_num=config.get("lstm_layer_num", 2),
                        dropout=config.get("dropout", 0.25),
                        teacher_forcing=config.get("teacher_forcing", False))
    learning_rate = config.get('lr', 1e-3)
    model.compile(optimizer=getattr(tf.keras.optimizers,
                                    config.get("optim", "Adam"))(learning_rate),
                  loss=config.get("loss", "mse"),
                  metrics=[config.get("metics", "mse")])
    return model
