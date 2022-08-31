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
import pickle
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import tensorflow.keras as keras

from bigdl.orca.automl.model.abstract import BaseModel
from bigdl.orca.automl.metrics import Evaluator


class LSTMSeq2Seq(BaseModel):

    def __init__(self, check_optional_config=False, future_seq_len=2):
        """
        Constructor of LSTM Seq2Seq model
        """
        self.model = None
        self.past_seq_len = None
        self.future_seq_len = future_seq_len
        self.feature_num = None
        self.target_col_num = None
        self.metric = None
        self.latent_dim = None
        self.batch_size = None
        self.check_optional_config = check_optional_config

    def _build_train(self, mc=False, **config):
        """
        build LSTM Seq2Seq model
        :param config:
        :return:
        """
        super()._check_config(**config)
        self.metric = config.get('metric', 'mean_squared_error')
        self.latent_dim = config.get('latent_dim', 128)
        self.dropout = config.get('dropout', 0.2)
        self.lr = config.get('lr', 0.001)
        # for restore in continuous training
        self.batch_size = config.get('batch_size', 64)
        training = True if mc else None

        # Define an input sequence and process it.
        self.encoder_inputs = Input(shape=(None, self.feature_num), name="encoder_inputs")
        encoder = LSTM(units=self.latent_dim,
                       dropout=self.dropout,
                       return_state=True,
                       name="encoder_lstm")
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs, training=training)
        # We discard `encoder_outputs` and only keep the states.
        self.encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = Input(shape=(None, self.target_col_num), name="decoder_inputs")
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        self.decoder_lstm = LSTM(self.latent_dim,
                                 dropout=self.dropout,
                                 return_sequences=True,
                                 return_state=True,
                                 name="decoder_lstm")
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs,
                                                  training=training,
                                                  initial_state=self.encoder_states)

        self.decoder_dense = Dense(self.target_col_num, name="decoder_dense")
        decoder_outputs = self.decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)
        self.model.compile(loss='mse',
                           metrics=[self.metric],
                           optimizer=keras.optimizers.RMSprop(lr=self.lr))
        return self.model

    def _restore_model(self):
        self.encoder_inputs = self.model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = self.model.layers[2].output  # lstm_1
        self.encoder_states = [state_h_enc, state_c_enc]

        self.decoder_inputs = self.model.input[1]  # input_2
        self.decoder_lstm = self.model.layers[3]

        self.decoder_dense = self.model.layers[4]

    def _build_inference(self, mc=False):
        training = True if mc else None
        # from our previous model - mapping encoder sequence to state vectors
        encoder_model = Model(self.encoder_inputs, self.encoder_states)

        # A modified version of the decoding stage that takes in predicted target inputs
        # and encoded state vectors, returning predicted target outputs and decoder state vectors.
        # We need to hang onto these state vectors to run the next step of the inference loop.
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = self.decoder_lstm(self.decoder_inputs,
                                                              training=training,
                                                              initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]

        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder_model = Model([self.decoder_inputs] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)
        return encoder_model, decoder_model

    def _decode_sequence(self, input_seq, mc=False):
        encoder_model, decoder_model = self._build_inference(mc=mc)
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((len(input_seq), 1, self.target_col_num))

        # Populate the first target sequence with end of encoding series value
        target_seq[:, 0] = input_seq[:, -1, :self.target_col_num]

        # Sampling loop for a batch of sequences - we will fill decoded_seq with predictions
        # (to simplify, here we assume a batch of size 1).

        decoded_seq = np.zeros((len(input_seq), self.future_seq_len, self.target_col_num))

        for i in range(self.future_seq_len):
            output, h, c = decoder_model.predict([target_seq] + states_value)

            decoded_seq[:, i] = output[:, 0]

            # Update the target sequence (of length 1).
            target_seq = np.zeros((len(input_seq), 1, self.target_col_num))
            target_seq[:, 0] = output[:, 0]

            # Update states
            states_value = [h, c]

        return decoded_seq

    def _get_decoder_inputs(self, x, y):
        """
        lagged target series for teacher forcing
        decoder_input data is one timestamp ahead of y
        :param x: 3-d array in format of (sample_num, past_sequence_len, feature_num)
        :param y: 3-d array in format of (sample_num, future_sequence_len, target_col_num)
                  Need to expand dimension if y is a 2-d array with one target col
        :return: 3-d array of decoder inputs
        """
        decoder_input_data = np.zeros(y.shape)
        decoder_input_data[1:, ] = y[:-1, ]
        decoder_input_data[0, 0] = x[-1, -1, :self.target_col_num]
        decoder_input_data[0, 1:] = y[0, :-1]

        return decoder_input_data

    def _get_len(self, x, y):
        self.past_seq_len = x.shape[1]
        self.feature_num = x.shape[2]
        # self.future_seq_len = y.shape[1]
        self.target_col_num = y.shape[2]

    def _expand_y(self, y):
        """
        expand dims for y.
        :param y:
        :return:
        """
        while len(y.shape) < 3:
            y = np.expand_dims(y, axis=2)
        return y

    def _pre_processing(self, x, y, validation_data):
        """
        pre_process input data.
        1. expand dims for y and val_y
        2. get decoder inputs for train data
        3. get decoder inputs for validation data
        :param x: train_x
        :param y: train_y
        :param validation_data:
        :return: network input
        """
        y = self._expand_y(y)
        self._get_len(x, y)
        decoder_input_data = self._get_decoder_inputs(x, y)
        if validation_data is not None:
            val_x, val_y = validation_data
            val_y = self._expand_y(val_y)
            val_decoder_input = self._get_decoder_inputs(val_x, val_y)
            validation_data = ([val_x, val_decoder_input], val_y)
        return x, y, decoder_input_data, validation_data

    def fit_eval(self, data, validation_data=None, mc=False, verbose=0, **config):
        """
        fit for one iteration
        :param data: could be a tuple with numpy ndarray with form (x, y)
        x: 3-d array in format (no. of samples, past sequence length, 2+feature length),
        in the last dimension, the 1st col is the time index (data type needs to be numpy datetime
        type, e.g. "datetime64"),
        the 2nd col is the target value (data type should be numeric)
        y: 2-d numpy array in format (no. of samples, future sequence length)
        if future sequence length > 1,
        or 1-d numpy array in format (no. of samples, ) if future sequence length = 1
        :param validation_data: tuple in format (x_test,y_test), data used for validation.
        If this is specified, validation result will be the optimization target for automl.
        Otherwise, train metric will be the optimization target.
        :param config: optimization hyper parameters
        :return: the resulting metric
        """
        x, y = data[0], data[1]
        x, y, decoder_input_data, validation_data = self._pre_processing(x, y, validation_data)

        # if model is not initialized, __build the model
        if self.model is None:
            self._build_train(mc=mc, **config)

        # batch_size = config.get('batch_size', 64)
        # lr = self.lr
        # name = "seq2seq-batch_size-{}-epochs-{}-lr-{}-time-{}"\
        #     .format(batch_size, epochs, lr, time())
        # tensorboard = TensorBoard(log_dir="logs/" + name)

        hist = self.model.fit([x, decoder_input_data], y,
                              validation_data=validation_data,
                              batch_size=self.batch_size,
                              epochs=config.get("epochs", 10),
                              verbose=verbose,
                              # callbacks=[tensorboard]
                              )
        # check input metric value
        hist_metric_name = keras.metrics.get(self.metric).__name__
        # model.metrics_names are available only after a keras model has been trained/evaluated
        compiled_metric_names = self.model.metrics_names.copy()
        print(compiled_metric_names)
        compiled_metric_names.remove("loss")
        if hist_metric_name in compiled_metric_names:
            metric_name = hist_metric_name
        elif self.metric in compiled_metric_names:
            metric_name = self.metric
        else:
            from bigdl.nano.utils.log4Error import invalidInputError
            invalidInputError(False,
                              f"Input metric in fit_eval should be one of the metrics that "
                              f"are used "
                              f"to compile the model. Got metric value of {self.metric} and "
                              f"the metrics in compile are {compiled_metric_names}")
        if validation_data is None:
            result = hist.history.get(metric_name)[-1]
        else:
            result = hist.history.get('val_' + metric_name)[-1]
        return {self.metric: result}

    def evaluate(self, x, y, metric=['mse']):
        """
        Evaluate on x, y
        :param x: input
        :param y: target
        :param metric: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        y_pred = self.predict(x)
        # y = np.squeeze(y, axis=2)
        if self.target_col_num == 1:
            return [Evaluator.evaluate(m, y, y_pred) for m in metric]
        else:
            return [np.array([Evaluator.evaluate(m, y[:, i, :], y_pred[:, i, :])
                              for i in range(self.future_seq_len)])
                    for m in metric]

    def predict(self, x, mc=False):
        """
        Prediction on x.
        :param x: input
        :return: predicted y (expected dimension = 2)
        """
        y_pred = self._decode_sequence(x, mc=mc)
        if self.target_col_num == 1:
            y_pred = np.squeeze(y_pred, axis=2)
        return y_pred

    def predict_with_uncertainty(self, x, n_iter=100):
        result = np.array([self.predict(x, mc=True) for i in range(n_iter)])
        prediction = result.mean(axis=0)
        uncertainty = result.var(axis=0)
        return prediction, uncertainty

    def state_dict(self):
        state = {
            "weights": self.model.get_weights(),
            "config": {"past_seq_len": self.past_seq_len,
                       "feature_num": self.feature_num,
                       "future_seq_len": self.future_seq_len,
                       "target_col_num": self.target_col_num,
                       "metric": self.metric,
                       "latent_dim": self.latent_dim,
                       "batch_size": self.batch_size}
        }
        return state

    def save(self, checkpoint_file, config_path=None):
        """
        save model to file.
        :param checkpoint_file: the model file path to be saved to.
        :param config_path: the config file path to be saved to.
        :return:
        """
        state_dict = self.state_dict()
        with open(checkpoint_file, "wb") as f:
            pickle.dump(state_dict, f)

    def restore(self, checkpoint_file, **config):
        """
        restore model from file
        :param checkpoint_file: the model file
        :param config: the trial config
        :return: the restored model
        """
        with open(checkpoint_file, "rb") as f:
            state_dict = pickle.load(f)
        config = state_dict["config"]

        self.past_seq_len = config["past_seq_len"]
        self.feature_num = config["feature_num"]
        self.future_seq_len = config["future_seq_len"]
        self.target_col_num = config["target_col_num"]
        self.metric = config["metric"]
        self.latent_dim = config["latent_dim"]
        self.batch_size = config["batch_size"]

        self._build_train(**config)
        self.model.set_weights(state_dict["weights"])
        self._restore_model()
        # self.model.load_weights(file_path)

    def _get_required_parameters(self):
        return {
            # 'input_shape_x',
            # 'input_shape_y',
            # 'out_units'
        }

    def _get_optional_parameters(self):
        return {
            'past_seq_len',
            'latent_dim',
            'dropout',
            'metric',
            'lr',
            'epochs',
            'batch_size'
        }
