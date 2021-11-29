#
# Copyright 2018 Alibaba Group. All Rights Reserved.
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
# Modifications copyright 2016 The BigDL Authors.
# ========================================================

import tensorflow as tf
from rnn import dynamic_rnn
from utils import *


class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr,
                 data_type='FP32', use_negsampling=False):
        tf.reset_default_graph()
        if data_type == 'FP32':
            self.model_dtype = tf.float32
        elif data_type == 'FP16':
            self.model_dtype = tf.float16
        else:
            raise ValueError("Invalid model data type: %s" % data_type)

        with tf.name_scope('Inputs'):
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cat_his_batch_ph')
            self.mask = tf.placeholder(self.model_dtype, [None, None], name='mask')
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.lr = lr
            self.use_negsampling = use_negsampling
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.cat_batch_ph = tf.placeholder(tf.int32, [None, ], name='cat_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None],
                                                         name='noclk_mid_batch_ph')
                # generate 3 item IDs from negative sampling.
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None],
                                                         name='noclk_cat_batch_ph')

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM],
                                                      dtype=self.model_dtype)
            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var,
                                                             self.uid_batch_ph)

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM],
                                                      dtype=self.model_dtype)

            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                             self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                 self.mid_his_batch_ph)
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                           self.noclk_mid_batch_ph)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM],
                                                      dtype=self.model_dtype)

            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var,
                                                             self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var,
                                                                 self.cat_his_batch_ph)
            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(
                    self.cat_embeddings_var, self.noclk_cat_batch_ph)

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat(
                [self.noclk_mid_his_batch_embedded[:, :, 0, :],
                 self.noclk_cat_his_batch_embedded[:, :, 0, :]], -1)
            # 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
            self.noclk_item_his_eb = tf.reshape(
                self.noclk_item_his_eb, [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1], 36])
            # cat embedding 18 concate item embedding 18.

            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded,
                                           self.noclk_cat_his_batch_embedded], -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

    def build_fcn_net(self, inp, use_dice=False):
        def dtype_getter(getter, name, dtype=None, *args, **kwargs):
            var = getter(name, dtype=self.model_dtype, *args, **kwargs)
            return var

        with tf.variable_scope("fcn", custom_getter=dtype_getter, dtype=self.model_dtype):
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
            dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
            if use_dice:
                dnn1 = dice(dnn1, name='dice_1', data_type=self.model_dtype)
            else:
                dnn1 = prelu(dnn1, 'prelu1')

            dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
            if use_dice:
                dnn2 = dice(dnn2, name='dice_2', data_type=self.model_dtype)
            else:
                dnn2 = prelu(dnn2, 'prelu2')
            dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
            self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

            with tf.name_scope("Metrics"):
                # Cross-entropy loss and optimizer initialization
                ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
                self.loss = ctr_loss
                if self.use_negsampling:
                    self.loss += self.aux_loss
                tf.summary.scalar('loss', self.loss)
                self.optim = tf.train.AdamOptimizer(learning_rate=self.lr)
                self.optimizer = self.optim.minimize(self.loss)
                # Accuracy metric
                self.accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), self.model_dtype))
                tf.summary.scalar('accuracy', self.accuracy)

            self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None):
        def dtype_getter(getter, name, dtype=None, *args, **kwargs):
            var = getter(name, dtype=self.model_dtype, *args, **kwargs)
            return var

        with tf.variable_scope("aux_loss", custom_getter=dtype_getter, dtype=self.model_dtype):
            mask = tf.cast(mask, self.model_dtype)
            click_input_ = tf.concat([h_states, click_seq], -1)
            noclick_input_ = tf.concat([h_states, noclick_seq], -1)
            click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
            noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
            click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
            noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_),
                                         [-1, tf.shape(noclick_seq)[1]]) * mask
            loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
            return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2,  2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat


class Model_DIN_V2_Gru_att_Gru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr,
                 data_type='FP32', use_negsampling=False):
        super(Model_DIN_V2_Gru_att_Gru, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       lr, data_type, use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE,
                                                    self.mask, softmax_stag=1, stag='1_1',
                                                    mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=att_outputs,
                                                     sequence_length=self.seq_len_ph,
                                                     dtype=tf.float32, scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum, final_state2], 1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)


class Model_DIN_V2_Gru_Gru_att(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr,
                 data_type='FP32', use_negsampling=False):
        super(Model_DIN_V2_Gru_Gru_att, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       lr, data_type, use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                          sequence_length=self.seq_len_ph,
                                          dtype=tf.float32, scope="gru2")
            tf.summary.histogram('GRU2_outputs', rnn_outputs2)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs2, ATTENTION_SIZE,
                                                    self.mask, softmax_stag=1, stag='1_1',
                                                    mode='LIST', return_alphas=True)
            att_fea = tf.reduce_sum(att_outputs, 1)
            tf.summary.histogram('att_fea', att_fea)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum, att_fea], 1)
        self.build_fcn_net(inp, use_dice=True)


class Model_WideDeep(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr,
                 data_type='FP32', use_negsampling=False):
        super(Model_WideDeep, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                             ATTENTION_SIZE, lr, data_type, use_negsampling)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        # Fully connected layer
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'p1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'p2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        d_layer_wide = tf.concat([tf.concat([self.item_eb, self.item_his_eb_sum], axis=-1),
                                 self.item_eb * self.item_his_eb_sum], axis=-1)
        d_layer_wide = tf.layers.dense(d_layer_wide, 2, activation=None, name='f_fm')
        self.y_hat = tf.nn.softmax(dnn3 + d_layer_wide)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            tf.summary.scalar('loss', self.loss)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.optimizer = self.optim.minimize(self.loss)
            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph),
                                                   self.model_dtype))
            tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()


class Model_DIN_V2_Gru_QA_attGru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr,
                 data_type='FP32', use_negsampling=False):
        super(Model_DIN_V2_Gru_QA_attGru, self).__init__(n_uid, n_mid, n_cat,
                                                         EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                         lr, data_type, use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE,
                                                    self.mask, softmax_stag=1, stag='1_1',
                                                    mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(QAAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores=tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph,
                                                     dtype=self.model_dtype, scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)


class Model_DNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr,
                 data_type='FP32', use_negsampling=False):
        super(Model_DNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE, lr, data_type, use_negsampling)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        self.build_fcn_net(inp, use_dice=False)


class Model_PNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr,
                 data_type='FP32', use_negsampling=False):
        super(Model_PNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE, lr, data_type, use_negsampling)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum], 1)

        # Fully connected layer
        self.build_fcn_net(inp, use_dice=False)


class Model_DIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr,
                 data_type='FP32', use_negsampling=False):
        super(Model_DIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE, lr, data_type, use_negsampling)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE,
                                             self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum, att_fea], -1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)


class Model_DIN_V2_Gru_Vec_attGru_Neg(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr,
                 data_type='FP32', use_negsampling=True):
        super(Model_DIN_V2_Gru_Vec_attGru_Neg, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM,
                                                              HIDDEN_SIZE, ATTENTION_SIZE,
                                                              lr, data_type, use_negsampling)

        def dtype_getter(getter, name, dtype=None, *args, **kwargs):
            var = getter(name, dtype=self.model_dtype, *args, **kwargs)
            return var

        with tf.variable_scope("dien", custom_getter=dtype_getter, dtype=self.model_dtype):
            # RNN layer(-s)
            with tf.name_scope('rnn_1'):
                rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                             sequence_length=self.seq_len_ph,
                                             dtype=self.model_dtype, scope="gru1")
                tf.summary.histogram('GRU_outputs', rnn_outputs)

            aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                             self.noclk_item_his_eb[:, 1:, :],
                                             self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            # Attention layer
            with tf.name_scope('Attention_layer_1'):
                att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE,
                                                        self.mask, softmax_stag=1, stag='1_1',
                                                        mode='LIST', return_alphas=True)
                tf.summary.histogram('alpha_outputs', alphas)

            with tf.name_scope('rnn_2'):
                rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE),
                                                         inputs=rnn_outputs,
                                                         att_scores=tf.expand_dims(alphas, -1),
                                                         sequence_length=self.seq_len_ph,
                                                         dtype=self.model_dtype, scope="gru2")
                tf.summary.histogram('GRU2_Final_State', final_state2)

            inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                             self.item_eb * self.item_his_eb_sum, final_state2], 1)
            self.build_fcn_net(inp, use_dice=True)


class Model_DIN_V2_Gru_Vec_attGru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr,
                 data_type='FP32', use_negsampling=False):
        super(Model_DIN_V2_Gru_Vec_attGru, self).__init__(n_uid, n_mid, n_cat,
                                                          EMBEDDING_DIM, HIDDEN_SIZE,
                                                          ATTENTION_SIZE, lr,
                                                          data_type, use_negsampling)

        use_dtype = tf.float32
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=self.model_dtype,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE,
                                                    self.mask, softmax_stag=1, stag='1_1',
                                                    mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores=tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph,
                                                     dtype=self.model_dtype,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)
