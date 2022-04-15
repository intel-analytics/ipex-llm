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
from tensorflow.keras import backend as K
from bigdl.nano.tf.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Lambda, Reshape, Layer


class Encoder(Layer):
    def __init__(self, past_seq_len, input_feature_num,
                 lstm_hidden_dim=128, lstm_layer_num=2, dropout=0.2):
        self.past_seq_len = past_seq_len
        self.input_feature_num = input_feature_num
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layer_num = lstm_layer_num
        self.dropout = dropout
        self.encoder_lstm = []
        for i in range(lstm_layer_num):
            self.encoder_lstm.append(LSTM(self.lstm_hidden_dim,
                                          return_sequences=True,
                                          input_shape=(past_seq_len, input_feature_num),
                                          return_state=True,
                                          dropout=dropout,
                                          name="cus_encoder"+str(i+1)))
        super(Encoder, self).__init__()

    def call(self, enc_inp):
        enc_states = None
        for encoder in self.encoder_lstm:
            enc_out, *enc_states = encoder(enc_inp, initial_state=enc_states)
            enc_inp = enc_out
        return enc_states

    def get_config(self):
        return {"past_seq_len": self.past_seq_len,
                "input_feature_num": self.input_feature_num,
                "lstm_hidden_dim": self.lstm_hidden_dim,
                "lstm_layer_num": self.lstm_layer_num,
                "dropout": self.dropout,
                "encoder_lstm": self.encoder_lstm}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Decoder(Layer):
    def __init__(self, output_feature_num, lstm_hidden_dim=128, lstm_layer_num=2, dropout=0.2):
        self.output_feature_num = output_feature_num
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layer_num = lstm_layer_num
        self.dropout = dropout
        self.decoder_lstm = []
        for i in range(lstm_layer_num):
            self.decoder_lstm.append(LSTM(self.lstm_hidden_dim,
                                          return_sequences=True,
                                          return_state=True,
                                          input_shape=(1, output_feature_num),
                                          dropout=dropout,
                                          name="cus_decoder"+str(i+1)))
        self.fc = Dense(self.output_feature_num)
        super(Decoder, self).__init__()

    def call(self, dec_inp, states):
        decoder_states = states
        for decoder in self.decoder_lstm:
            dec_out, *decoder_states = decoder(dec_inp, initial_state=decoder_states)
            dec_inp = dec_out
        return dec_out

    def get_config(self):
        return {"output_feature_num": self.output_feature_num,
                "lstm_hidden_dim": self.lstm_hidden_dim,
                "lstm_layer_num": self.lstm_layer_num,
                "dropout": self.dropout,
                "decoder_lstm": self.decoder_lstm}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LSTMSeq2Seq(Model):
    def __init__(self,
                 past_seq_len,
                 future_seq_len,
                 input_feature_num,
                 output_feature_num,
                 lstm_hidden_dim=128,
                 lstm_layer_num=2,
                 dropout=0.2,
                 teacher_forcing=False):
        super(LSTMSeq2Seq, self).__init__()
        self.past_seq_len = past_seq_len
        self.future_seq_len = future_seq_len
        self.input_feature_num = input_feature_num
        self.output_feature_num = output_feature_num
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layer_num = lstm_layer_num
        self.dropout = dropout
        self.teacher_forcing = teacher_forcing
        self.decoder_inputs = Reshape((1, output_feature_num),
                                      input_shape=(output_feature_num,))
        self.encoder = Encoder(past_seq_len, input_feature_num,
                               lstm_hidden_dim, lstm_layer_num, dropout)
        self.decoder = Decoder(output_feature_num, lstm_hidden_dim, lstm_layer_num, dropout)
        self.fc = Dense(output_feature_num)

    def call(self, inp, target_seq=None, training=True):
        decoder_inp = inp
        states = self.encoder(inp, training=training)
        decoder_inputs = self.decoder_inputs(decoder_inp[:, -1, :self.output_feature_num])
        all_outputs = []
        for seq_len in range(self.future_seq_len):
            if self.teacher_forcing and target_seq is not None:
                decoder_inputs = target_seq[:, seq_len:seq_len+1, :]
            dec_outputs = self.decoder(decoder_inputs, states)
            decoder_outputs = self.fc(dec_outputs)
            all_outputs.append(decoder_outputs)
        outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
        return outputs

    def get_config(self):
        return {"past_seq_len": self.past_seq_len,
                "future_seq_len": self.future_seq_len,
                "input_feature_num": self.input_feature_num,
                "output_feature_num": self.output_feature_num,
                "lstm_hidden_dim": self.lstm_hidden_dim,
                "lstm_layer_num": self.lstm_layer_num,
                "dropout": self.dropout,
                "teacher_forcing": self.teacher_forcing}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def model_creator(config):
    model = LSTMSeq2Seq(input_feature_num=config["input_feature_num"],
                        output_feature_num=config["output_feature_num"],
                        past_seq_len=config["past_seq_len"],
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
