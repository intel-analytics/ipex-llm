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

# Original imports
import time
import argparse
import tensorflow as tf
import os
import sys
import math
import collections
from tensorflow.python.client import timeline
import json

from tensorflow.python.ops import partitioned_variables

# Extra imports
import ray
from bigdl.dllib.utils.log4Error import *
from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.orca.data.utils import process_spark_xshards
from bigdl.orca.learn.utils import maybe_dataframe_to_xshards
from bigdl.friesian.feature import FeatureTable

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))

# Definition of some constants
CONTINUOUS_COLUMNS = ['I' + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ['C' + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ['clicked']
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
HASH_BUCKET_SIZES = {
    'C1': 2500,
    'C2': 2000,
    'C3': 300000,
    'C4': 250000,
    'C5': 1000,
    'C6': 100,
    'C7': 20000,
    'C8': 4000,
    'C9': 20,
    'C10': 100000,
    'C11': 10000,
    'C12': 250000,
    'C13': 40000,
    'C14': 100,
    'C15': 100,
    'C16': 200000,
    'C17': 50,
    'C18': 10000,
    'C19': 4000,
    'C20': 20,
    'C21': 250000,
    'C22': 100,
    'C23': 100,
    'C24': 250000,
    'C25': 400,
    'C26': 100000
}

EMBEDDING_DIMENSIONS = {
    'C1': 64,
    'C2': 64,
    'C3': 128,
    'C4': 128,
    'C5': 64,
    'C6': 64,
    'C7': 64,
    'C8': 64,
    'C9': 64,
    'C10': 128,
    'C11': 64,
    'C12': 128,
    'C13': 64,
    'C14': 64,
    'C15': 64,
    'C16': 128,
    'C17': 64,
    'C18': 64,
    'C19': 64,
    'C20': 64,
    'C21': 128,
    'C22': 64,
    'C23': 64,
    'C24': 128,
    'C25': 64,
    'C26': 128
}


class WDL():
    def __init__(self,
                 wide_column=None,
                 deep_column=None,
                 dnn_hidden_units=[1024, 512, 256],
                 optimizer_type='adam',
                 linear_learning_rate=0.2,
                 deep_learning_rate=0.01,
                 inputs=None,
                 bf16=False,
                 stock_tf=None,
                 adaptive_emb=False,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not inputs:
            invalidInputError(False, "Dataset is not defined.")
        self._feature = inputs[0]
        self._label = inputs[1]

        self._wide_column = wide_column
        self._deep_column = deep_column
        if not wide_column or not deep_column:
            invalidInputError(False, "Wide column or Deep column is not defined.")

        self.tf = stock_tf
        self.bf16 = False if self.tf else bf16
        self.is_training = True
        self._adaptive_emb = adaptive_emb

        self._dnn_hidden_units = dnn_hidden_units
        self._linear_learning_rate = linear_learning_rate
        self._deep_learning_rate = deep_learning_rate
        self._optimizer_type = optimizer_type
        self._input_layer_partitioner = input_layer_partitioner
        self._dense_layer_partitioner = dense_layer_partitioner

        self._create_model()
        with tf.name_scope('head'):
            self._create_loss()
            self._create_optimizer()
            self._create_metrics()

    # used to add summary in tensorboard
    def _add_layer_summary(self, value, tag):
        tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                          tf.nn.zero_fraction(value))
        tf.summary.histogram('%s/activation' % tag, value)

    def _dnn(self, dnn_input, dnn_hidden_units=None, layer_name=''):
        for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
            with tf.variable_scope(layer_name + '_%d' % layer_id,
                                   partitioner=self._dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE) as dnn_layer_scope:
                dnn_input = tf.layers.dense(
                    dnn_input,
                    units=num_hidden_units,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name=dnn_layer_scope)

                self._add_layer_summary(dnn_input, dnn_layer_scope.name)
        return dnn_input

    # create model
    def _create_model(self):
        # Dnn part
        with tf.variable_scope('dnn'):
            # input layer
            with tf.variable_scope('input_from_feature_columns',
                                   partitioner=self._input_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                if self._adaptive_emb and not self.tf:
                    '''Adaptive Embedding Feature Part 1 of 2'''
                    adaptive_mask_tensors = {}
                    for col in CATEGORICAL_COLUMNS:
                        adaptive_mask_tensors[col] = tf.ones([args.batch_size],
                                                             tf.int32)
                    net = tf.feature_column.input_layer(
                        features=self._feature,
                        feature_columns=self._deep_column,
                        adaptive_mask_tensors=adaptive_mask_tensors)
                else:
                    net = tf.feature_column.input_layer(
                        features=self._feature,
                        feature_columns=self._deep_column)
                self._add_layer_summary(net, 'input_from_feature_columns')

            # hidden layers
            dnn_scope = tf.variable_scope(
                'dnn_layers', partitioner=self._dense_layer_partitioner, reuse=tf.AUTO_REUSE)
            with dnn_scope.keep_weights(dtype=tf.float32) if self.bf16 else dnn_scope:
                if self.bf16:
                    net = tf.cast(net, dtype=tf.bfloat16)

                net = self._dnn(net, self._dnn_hidden_units, 'hiddenlayer')

                if self.bf16:
                    net = tf.cast(net, dtype=tf.float32)

                # dnn logits
                logits_scope = tf.variable_scope('logits')
                with logits_scope.keep_weights(dtype=tf.float32) if self.bf16 \
                        else logits_scope as dnn_logits_scope:
                    dnn_logits = tf.layers.dense(net,
                                                 units=1,
                                                 activation=None,
                                                 name=dnn_logits_scope)
                    self._add_layer_summary(dnn_logits, dnn_logits_scope.name)

        # linear part
        with tf.variable_scope(
                'linear', partitioner=self._dense_layer_partitioner) as scope:
            linear_logits = tf.feature_column.linear_model(
                units=1,
                features=self._feature,
                feature_columns=self._wide_column,
                sparse_combiner='sum',
                weight_collections=None,
                trainable=True)

            self._add_layer_summary(linear_logits, scope.name)

        self._logits = tf.add_n([dnn_logits, linear_logits])
        self.probability = tf.math.sigmoid(self._logits)
        self.output = tf.round(self.probability)

    # compute loss
    def _create_loss(self):
        self._logits = tf.squeeze(self._logits)
        self.loss = tf.losses.sigmoid_cross_entropy(
            self._label,
            self._logits,
            scope='loss',
            reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        tf.summary.scalar('loss', self.loss)

    # define optimizer and generate train_op
    def _create_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        if self.tf or self._optimizer_type == 'adam':
            dnn_optimizer = tf.train.AdamOptimizer(
                learning_rate=self._deep_learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
        elif self._optimizer_type == 'adagrad':
            dnn_optimizer = tf.train.AdagradOptimizer(
                learning_rate=self._deep_learning_rate,
                initial_accumulator_value=0.1,
                use_locking=False)
        elif self._optimizer_type == 'adamasync':
            dnn_optimizer = tf.train.AdamAsyncOptimizer(
                learning_rate=self._deep_learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
        elif self._optimizer_type == 'adagraddecay':
            dnn_optimizer = tf.train.AdagradDecayOptimizer(
                learning_rate=self._deep_learning_rate,
                global_step=self.global_step)
        else:
            invalidInputError(False, "Optimzier type error.")

        linear_optimizer = tf.train.FtrlOptimizer(
            learning_rate=self._linear_learning_rate,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0)
        train_ops = []
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_ops.append(
                dnn_optimizer.minimize(self.loss,
                                       var_list=tf.get_collection(
                                           tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope='dnn'),
                                       global_step=self.global_step))
            train_ops.append(
                linear_optimizer.minimize(self.loss,
                                          var_list=tf.get_collection(
                                              tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope='linear')))
            self.train_op = tf.group(*train_ops)

    # compute acc & auc
    def _create_metrics(self):
        self.acc, self.acc_op = tf.metrics.accuracy(labels=self._label,
                                                    predictions=self.output)
        self.auc, self.auc_op = tf.metrics.auc(labels=self._label,
                                               predictions=self.probability,
                                               num_thresholds=1000)
        tf.summary.scalar('eval_acc', self.acc)
        tf.summary.scalar('eval_auc', self.auc)


# generate feature columns
def build_feature_columns():
    # Notes: Statistics of Kaggle's Criteo Dataset has been calculated in advance to save time.
    mins_list = [
        0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    range_list = [
        1539.0, 22069.0, 65535.0, 561.0, 2655388.0, 233523.0, 26297.0, 5106.0,
        24376.0, 9.0, 181.0, 1807.0, 6879.0
    ]

    def make_minmaxscaler(min, range):
        def minmaxscaler(col):
            return (col - min) / range

        return minmaxscaler

    deep_columns = []
    wide_columns = []
    for column_name in FEATURE_COLUMNS:
        if column_name in CATEGORICAL_COLUMNS:
            categorical_column = tf.feature_column.categorical_column_with_identity(
                column_name, num_buckets=10000)
            # categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
            #      column_name, hash_bucket_size=10000, dtype=tf.int64)
            wide_columns.append(categorical_column)

            if not args.tf:
                '''Feature Elimination of EmbeddingVariable Feature'''
                if args.ev_elimination == 'gstep':
                    # Feature elimination based on global steps
                    evict_opt = tf.GlobalStepEvict(steps_to_live=4000)
                elif args.ev_elimination == 'l2':
                    # Feature elimination based on l2 weight
                    evict_opt = tf.L2WeightEvict(l2_weight_threshold=1.0)
                else:
                    evict_opt = None
                '''Feature Filter of EmbeddingVariable Feature'''
                if args.ev_filter == 'cbf':
                    # CBF-based feature filter
                    filter_option = tf.CBFFilter(
                        filter_freq=3,
                        max_element_size=2**30,
                        false_positive_probability=0.01,
                        counter_type=tf.int64)
                elif args.ev_filter == 'counter':
                    # Counter-based feature filter
                    filter_option = tf.CounterFilter(filter_freq=3)
                else:
                    filter_option = None
                ev_opt = tf.EmbeddingVariableOption(
                    evict_option=evict_opt, filter_option=filter_option)

                if args.ev:
                    '''Embedding Variable Feature'''
                    categorical_column = tf.feature_column.categorical_column_with_embedding(
                        column_name, dtype=tf.int64, ev_option=ev_opt)
                elif args.adaptive_emb:
                    '''                 Adaptive Embedding Feature Part 2 of 2
                    Except the follow code, a dict, 'adaptive_mask_tensors', is need as the input of
                    'tf.feature_column.input_layer(adaptive_mask_tensors=adaptive_mask_tensors)'.
                    For column 'COL_NAME',the value of adaptive_mask_tensors['$COL_NAME'] is a int32
                    tensor with shape [batch_size].
                    '''
                    categorical_column = \
                        tf.feature_column.categorical_column_with_adaptive_embedding(
                            column_name,
                            hash_bucket_size=HASH_BUCKET_SIZES[column_name],
                            dtype=tf.int64,
                            ev_option=ev_opt)
                elif args.dynamic_ev:
                    '''Dynamic-dimension Embedding Variable'''
                    print(
                        "Dynamic-dimension Embedding Variable isn't really enabled in model."
                    )
                    sys.exit()

            if args.tf or not args.emb_fusion:
                embedding_column = tf.feature_column.embedding_column(
                    categorical_column,
                    dimension=EMBEDDING_DIMENSIONS[column_name],
                    combiner='mean')
            else:
                '''Embedding Fusion Feature'''
                embedding_column = tf.feature_column.embedding_column(
                    categorical_column,
                    dimension=EMBEDDING_DIMENSIONS[column_name],
                    combiner='mean',
                    do_fusion=args.emb_fusion)

            deep_columns.append(embedding_column)
        else:
            normalizer_fn = None
            i = CONTINUOUS_COLUMNS.index(column_name)
            normalizer_fn = make_minmaxscaler(mins_list[i], range_list[i])
            column = tf.feature_column.numeric_column(
                column_name, normalizer_fn=normalizer_fn, shape=(1, ))
            wide_columns.append(column)
            deep_columns.append(column)

    return wide_columns, deep_columns


def train(sess_config,
          input_hooks,
          model,
          data_init_op,
          config,
          tf_config=None,
          server=None):
    steps = config["train_steps"]
    checkpoint_dir = config["checkpoint_dir"]

    model.is_training = True
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.local_variables_initializer(), data_init_op),
        saver=tf.train.Saver(max_to_keep=config["keep_checkpoint_max"]))

    stop_hook = tf.train.StopAtStepHook(last_step=steps)
    log_hook = tf.train.LoggingTensorHook(
        {
            'steps': model.global_step,
            'loss': model.loss
        }, every_n_iter=100)
    hooks.append(stop_hook)
    hooks.append(log_hook)
    if config["timeline"] > 0:
        hooks.append(
            tf.train.ProfilerHook(save_steps=config["timeline"],
                                  output_dir=checkpoint_dir))
    save_steps = config["save_steps"] if config["save_steps"] or config["no_eval"] else steps
    '''
                            Incremental_Checkpoint
    Please add `save_incremental_checkpoint_secs` in 'tf.train.MonitoredTrainingSession'
    it's default to None, Incremental_save checkpoint time in seconds can be set
    to use incremental checkpoint function, like `tf.train.MonitoredTrainingSession(
        save_incremental_checkpoint_secs=args.incremental_ckpt)`
    '''
    if config["incremental_ckpt"] and not config["tf"]:
        print("Incremental_Checkpoint is not really enabled.")
        print("Please see the comments in the code.")
        sys.exit()

    print('Creating session')
    t = time.time()
    with tf.train.MonitoredTrainingSession(
            master=server.target if server else '',
            is_chief=tf_config['is_chief'] if tf_config else True,
            hooks=hooks,
            scaffold=scaffold,
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_steps=save_steps,
            summary_dir=checkpoint_dir,
            save_summaries_steps=config["save_steps"],
            config=sess_config) as sess:
        print(f'Session creation time: {time.time() - t:.8f}s')
        while not sess.should_stop():
            sess.run([model.loss, model.train_op])
    print("Training completed.")


def eval(sess_config, input_hooks, model, data_init_op, steps, checkpoint_dir):
    model.is_training = False
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.local_variables_initializer(), data_init_op))
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold, checkpoint_dir=checkpoint_dir, config=sess_config)
    writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'eval'))
    merged = tf.summary.merge_all()
    with tf.train.MonitoredSession(session_creator=session_creator,
                                   hooks=hooks) as sess:
        for _in in range(1, steps + 1):
            if (_in != steps):
                sess.run([model.acc_op, model.auc_op])
                if (_in % 1000 == 0):
                    print("Evaluation complete:[{}/{}]".format(_in, steps))
            else:
                eval_acc, eval_auc, events = sess.run(
                    [model.acc_op, model.auc_op, merged])
                writer.add_summary(events, _in)
                print("Evaluation complete:[{}/{}]".format(_in, steps))
                print("ACC = {}\nAUC = {}".format(eval_acc, eval_auc))

    return eval_acc, eval_auc


def main(train_dataset, test_dataset=None, tf_config=None, server=None, config=None):
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    # create feature column
    wide_column, deep_column = build_feature_columns()

    # create variable partitioner for distributed training
    num_ps_replicas = len(tf_config['ps_hosts']) if tf_config else 0
    input_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas,
        min_slice_size=config["input_layer_partitioner"] <<
        20) if config["input_layer_partitioner"] else None
    dense_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas,
        min_slice_size=config["dense_layer_partitioner"] <<
        10) if config["dense_layer_partitioner"] else None

    # Session config
    sess_config = tf.ConfigProto()
    sess_config.inter_op_parallelism_threads = config["inter"]
    sess_config.intra_op_parallelism_threads = config["intra"]

    # Session hooks
    hooks = []

    if config["smartstaged"] and not config["tf"]:
        '''Smart staged Feature'''
        next_element = tf.staged(next_element, num_threads=4, capacity=40)
        sess_config.graph_options.optimizer_options.do_smart_stage = True
        hooks.append(tf.make_prefetch_hook())
    if config["op_fusion"] and not config["tf"]:
        '''Auto Graph Fusion'''
        sess_config.graph_options.optimizer_options.do_op_fusion = True
    if config["micro_batch"] and not config["tf"]:
        '''Auto Mirco Batch'''
        sess_config.graph_options.optimizer_options.micro_batch_num = config["micro_batch"]

    # create model
    model = WDL(wide_column=wide_column,
                deep_column=deep_column,
                linear_learning_rate=config["linear_learning_rate"],
                deep_learning_rate=config["deep_learning_rate"],
                optimizer_type=config["optimizer"],
                bf16=config["bf16"],
                stock_tf=config["tf"],
                adaptive_emb=config["adaptive_emb"],
                inputs=next_element,
                input_layer_partitioner=input_layer_partitioner,
                dense_layer_partitioner=dense_layer_partitioner)

    # Run model training and evaluation
    train(sess_config, hooks, model, train_init_op, config, tf_config, server)
    # TODO: the original script won't evaluate in distributed mode, has issue in evaluation results
    if not config["no_eval"]:
        eval_acc, eval_auc = eval(sess_config, hooks, model, test_init_op, config['test_steps'],
                                  config['checkpoint_dir'])
        return eval_acc, eval_auc
    return None


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {'false', 'true'}:
        invalidInputError(False, 'Not a valid boolean string')
    return low_string == 'true'


# Get parse
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')
    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=0)
    parser.add_argument('--batch_size',
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=512)
    parser.add_argument('--output_dir',
                        help='Full path to model output directory. \
                            Default to ./result. Covered by --checkpoint. ',
                        required=False,
                        default='./result')
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output. \
                            Default to ./result/$MODEL_TIMESTAMP',
                        required=False)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=0)
    parser.add_argument('--seed',
                        help='set the random seed for tensorflow',
                        type=int,
                        default=2021)
    parser.add_argument('--optimizer',
                        type=str,
                        choices=['adam', 'adamasync', 'adagraddecay', 'adagrad'],
                        default='adamasync')
    parser.add_argument('--linear_learning_rate',
                        help='Learning rate for linear model',
                        type=float,
                        default=0.2)
    parser.add_argument('--deep_learning_rate',
                        help='Learning rate for deep model',
                        type=float,
                        default=0.01)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline. Default 0',
                        type=int,
                        default=0)
    parser.add_argument('--protocol',
                        type=str,
                        choices=['grpc', 'grpc++', 'star_server'],
                        default='grpc')
    parser.add_argument('--inter',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--intra',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--input_layer_partitioner',
                        help='slice size of input layer partitioner, units MB. Default 8MB',
                        type=int,
                        default=8)
    parser.add_argument('--dense_layer_partitioner',
                        help='slice size of dense layer partitioner, units KB. Default 16KB',
                        type=int,
                        default=16)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset.',
                        action='store_true')
    parser.add_argument('--tf',
                        help='Use TF 1.15.5 API and disable DeepRec feature to run a baseline.',
                        action='store_true')
    parser.add_argument('--smartstaged',
                        help='Whether to enable smart staged feature of DeepRec, Default to True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--emb_fusion',
                        help='Whether to enable embedding fusion, Default to True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--ev',
                        help='Whether to enable DeepRec EmbeddingVariable. Default False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--ev_elimination',
                        help='Feature Elimination of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'l2', 'gstep'],
                        default=None)
    parser.add_argument('--ev_filter',
                        help='Feature Filter of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'counter', 'cbf'],
                        default=None)
    parser.add_argument('--op_fusion',
                        help='Whether to enable Auto graph fusion feature. Default to True',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--micro_batch',
                        help='Set num for Auto Mirco Batch. Default close.',
                        type=int,
                        default=0)  # TODO: Default to True
    parser.add_argument('--adaptive_emb',
                        help='Whether to enable Adaptive Embedding. Default to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--dynamic_ev',
                        help='Whether to enable Dynamic-dimension Embedding Variable. '
                             'Default to False.',
                        type=boolean_string,
                        default=False)  # TODO: enable
    parser.add_argument('--incremental_ckpt',
                        help='Set time of save Incremental Checkpoint. Default 0 to close.',
                        type=int,
                        default=0)
    parser.add_argument('--workqueue',
                        help='Whether to enable Work Queue. Default to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--cluster_mode',
                        help='The cluster mode, such as local, k8s and yarn.',
                        type=str, default="local")
    parser.add_argument('--num_nodes',
                        help='The number of nodes to use in the cluster.',
                        type=int, default=1)
    parser.add_argument('--cores',
                        help='The number of cpu cores to use on each node.',
                        type=int, default=8)
    parser.add_argument('--instances_per_node',
                        help='The number of ps and worker instances to run on each node.',
                        type=int, default=1)
    parser.add_argument('--master',
                        help='k8s master ip and port.',
                        type=str, default=None)
    parser.add_argument('--num_ps',
                        help='The number of parameter servers to use.',
                        type=int, default=1)
    parser.add_argument('--in_memory',
                        help='Whether to run the example based on in-memory data ingestion.',
                        action='store_true')
    return parser


# Some DeepRec's features are enabled by ENV.
# This func is used to set ENV and enable these features.
# A triple quotes comment is used to introduce these features and play an emphasizing role.
def set_env_for_DeepRec():
    '''
    Set some ENV for these DeepRec's features enabled by ENV.
    More Detail information is shown in
    https://deeprec.readthedocs.io/zh/latest/index.html.
    START_STATISTIC_STEP & STOP_STATISTIC_STEP:
        On CPU platform, DeepRec supports memory optimization
        in both stand-alone and distributed training. It's default to open, and the
        default start and stop steps of collection is 1000 and 1100. Reduce the initial
        cold start time by the following settings.
    MALLOC_CONF: On CPU platform, DeepRec can use memory optimization with the jemalloc library.
        Please preload libjemalloc.so by `LD_PRELOAD=./libjemalloc.so.2 python ...`
    '''
    os.environ['START_STATISTIC_STEP'] = '100'
    os.environ['STOP_STATISTIC_STEP'] = '110'
    os.environ['MALLOC_CONF'] = \
        'background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000'


def find_free_port():
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class RayDeepRecCluster:
    def __init__(self, config=None, instances_per_node=1, num_ps=1, protocol="grpc"):
        self.config = config
        self.num_ps = num_ps
        self.protocol = protocol
        ray_ctx = OrcaContext.get_ray_context()
        cores_per_instance = ray_ctx.ray_node_cpu_cores // instances_per_node
        self.num_instances = ray_ctx.num_ray_nodes * instances_per_node
        self.num_workers = self.num_instances - self.num_ps
        num_chief = 1  # Normally there would always be only one chief.
        invalidInputError(
            self.num_instances >= 2,
            "There should be at least two instances, one parameter server and one worker")
        invalidInputError(self.num_workers > 0,
                          "Do not have enough resources to launch {} parameter servers. "
                          "Try to reduce num_ps".format(self.num_ps))

        RemoteRunner = ray.remote(num_cpus=cores_per_instance)(RayWorker)
        self.remote_ps = [RemoteRunner.remote(task_index=i, task_type="ps", config=self.config)
                          for i in range(num_ps)]
        chief = RemoteRunner.remote(task_index=0, task_type="chief", config=self.config)
        workers = [RemoteRunner.remote(task_index=i, task_type="worker", config=self.config)
                   for i in range(0, self.num_workers - num_chief)]
        self.remote_workers = [chief] + workers
        self.remote_instances = self.remote_ps + self.remote_workers
        self.setup_workers()

    def fit(self, train_df, test_df=None, in_memory=False, feature_cols=None,
            label_cols=None):
        # ps has already be launched and waiting and thus it is removed when training
        # as ps should not consume data.
        print("Train data partitions:", train_df.rdd.getNumPartitions())
        print("Test data partitions:", test_df.rdd.getNumPartitions())
        if train_df.rdd.getNumPartitions() < self.num_workers:
            train_df = train_df.repartition(self.num_workers)
        if test_df and test_df.rdd.getNumPartitions() < self.num_workers:
            test_df = test_df.repartition(self.num_workers)
        if not in_memory:
            train_sizes = train_df.rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
            train_processed_folder = self.config['data_location'] + "/train_processed"
            train_df.write.csv(path=train_processed_folder, mode="overwrite", header=False, sep=",")
            train_files_dict = self.divide_files(train_processed_folder, train_sizes)
            test_files_dict = None
            if test_df:
                test_sizes = test_df.rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
                test_processed_folder = \
                    self.config['data_location'] + "/test_processed"
                test_df.write.csv(
                    path=test_processed_folder, mode="overwrite", header=False, sep=",")
                test_files_dict = self.divide_files(test_processed_folder, test_sizes)
            worker_stats = ray.get([worker.step_file.remote(train_files_dict, test_files_dict)
                                    for worker in self.remote_workers])
        else:
            train_dataset, test_dataset = maybe_dataframe_to_xshards(train_df, test_df,
                                                                     feature_cols, label_cols,
                                                                     mode="fit",
                                                                     num_workers=self.num_workers,
                                                                     accept_str_col=True)
            ray_xshards = process_spark_xshards(train_dataset, self.num_workers)
            val_ray_xshards = None
            if test_df:
                val_ray_xshards = process_spark_xshards(test_dataset, self.num_workers)
            # Train
            worker_stats = self.fit_ray_xshards(ray_xshards, val_ray_xshards)
        print('Eval completed, eval acc,auc for each worker:', worker_stats)

    def fit_ray_xshards(self, train_shards, val_shards):
        if val_shards is None:
            # TODO: split data_creator from step
            def transform_func(worker, partition_refs):
                return worker.step.remote(partition_refs)

            worker_stats = train_shards.reduce_partitions_for_actors(self.remote_workers,
                                                                     transform_func)
        else:
            def zip_func(worker, this_partition_refs, that_partition_refs):
                return worker.step.remote(this_partition_refs, that_partition_refs)

            worker_stats = train_shards.zip_reduce_shards_with_actors(val_shards,
                                                                      self.remote_workers,
                                                                      zip_func)
        return worker_stats

    def divide_files(self, folder, sizes):
        import glob

        # Each file is of format: /path/to/processed/part-id-***.csv
        files = glob.glob(folder + "/*.csv")
        file_with_sizes = [(file, sizes[int(file.split("/")[-1].split("-")[1])])for file in files]
        num_files_per_worker = len(files) // self.num_workers
        num_remain_files = len(files) % self.num_workers
        extra_files = []
        if num_remain_files > 0:
            extra_files = file_with_sizes[-num_remain_files:]
        files_dict = dict()
        for worker in self.remote_workers:
            index = ray.get(worker.get_task_index.remote())
            worker_files = \
                file_with_sizes[index*num_files_per_worker:(index+1)*num_files_per_worker]
            if extra_files:
                worker_files += [extra_files.pop()]
            files_dict[index] = worker_files
        # key is worker id; value is list of (file, size) tuple.
        return files_dict

    def get_ps(self):
        return self.remote_ps

    def get_workers(self):
        return self.remote_workers

    def get_instances(self):
        return self.remote_instances

    def setup_workers(self):
        ray.get([worker.setup_address.remote() for worker in self.remote_instances])
        ps_ips = []
        chief_ips = []
        worker_ips = []
        for worker in self.remote_instances:
            ip = ray.get(worker.get_address.remote())
            task_type = ray.get(worker.get_task_type.remote())
            if task_type == 'ps':
                ps_ips.append(ip)
            elif task_type == 'chief':
                chief_ips.append(ip)
            else:
                worker_ips.append(ip)

        cluster_info = {
            "ps": ps_ips,
            "chief": chief_ips,
            "worker": worker_ips
        }
        print(cluster_info)
        ps_chief_refs = [worker.setup_distributed.remote(cluster_info, self.protocol)
                         for worker in self.remote_ps]
        ps_chief_refs += [self.remote_workers[0]
                              .setup_distributed.remote(cluster_info, self.protocol)]
        # Use ray.wait since ps server.join process won't terminate by itself and
        # thus using ray.get on all results would hang.
        # By using ray.wait, we wait for the chief to finish the distributed setting and proceed.
        # In this case, we don't need to manually kill the ps process at the very end.
        # This also tries to make sure that ps and chief is launched earlier than ordinary workers.
        finished, unfinished = ray.wait(ps_chief_refs, num_returns=1)
        print(ray.get(finished))
        worker_refs = [worker.setup_distributed.remote(cluster_info, self.protocol)
                       for worker in self.remote_workers[1:]]
        print(ray.get(worker_refs))


class RayWorker:
    def __init__(self, task_index, task_type, config):
        self.task_index = task_index
        self.task_type = task_type
        self.config = config
        # set fixed random seed
        tf.set_random_seed(config["seed"])
        logging.basicConfig(level=logging.INFO)

    def get_task_type(self):
        return self.task_type

    def get_task_index(self):
        return self.task_index

    def setup_distributed(self, cluster_config, protocol="grpc"):
        self.cluster_config = cluster_config

        ps_hosts = self.cluster_config["ps"]
        chief_hosts = self.cluster_config["chief"]
        worker_hosts = self.cluster_config["worker"]
        if chief_hosts:
            worker_hosts = chief_hosts + worker_hosts

        task_type = self.task_type
        if task_type == 'worker' and chief_hosts:
            self.task_index += 1

        if self.task_type == 'chief':
            task_type = 'worker'
            # Make sure ps is launched earlier than the chief
            # TODO: any better way to do this?
            time.sleep(2)

        print("ps hosts: ", ps_hosts)
        print("worker hosts: ", worker_hosts)
        print("task type: ", task_type)
        print("task index: ", self.task_index)
        cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
        self.server = tf.distribute.Server(cluster,
                                           job_name=task_type,
                                           task_index=self.task_index,
                                           protocol=protocol)
        if self.task_type == "ps":
            print("Launching parameter server")
            self.server.join()
        else:
            print("Launching worker")
            self.config["task_index"] = self.task_index
            self.tf_config = {
                'ps_hosts': ps_hosts,
                'worker_hosts': worker_hosts,
                'type': task_type,
                'index': self.task_index,
                'is_chief': self.task_type == "chief"
            }
            tf_device = tf.device(
                tf.train.replica_device_setter(
                    worker_device='/job:worker/task:%d' % self.task_index,
                    cluster=cluster))
        return self.task_type

    def step(self, data_refs, validation_data_refs=None):
        from bigdl.orca.data.utils import partition_get_data_label

        partition_list = ray.get(data_refs)
        partition_data = [item for partition in partition_list for item in partition]
        data, label = partition_get_data_label(
            partition_data, allow_tuple=True, allow_list=False)
        data_size = len(label)
        steps = data_size // self.config["batch_size"] + 1
        print("Number of train records for this worker: ", data_size)
        print("Number of steps for this worker: ", steps)
        train_dataset = to_tensor_slice_dataset(data, label, self.config)

        test_dataset = None
        if validation_data_refs:
            validation_partition_list = ray.get(validation_data_refs)
            validation_partition_data = \
                [item for partition in validation_partition_list for item in partition]
            validation_data, validation_label = partition_get_data_label(
                validation_partition_data, allow_tuple=True, allow_list=False)
            test_data_size = len(validation_label)
            test_steps = test_data_size // self.config["batch_size"] + 1
            self.config["test_steps"] = test_steps
            print("Number of test records for this worker: ", test_data_size)
            print("Number of test steps for this worker: ", self.config["test_steps"])
            test_dataset = to_tensor_slice_dataset(validation_data, validation_label, self.config)

        return main(train_dataset, test_dataset, self.tf_config, self.server, self.config)

    def step_file(self, train_files, test_files=None):
        train_dataset, data_size = to_textline_dataset(
            train_files[self.task_index], self.config)
        steps = data_size // self.config["batch_size"] + 1
        print("Number of train records for this worker: ", data_size)
        print("Number of steps for this worker: ", steps)
        test_dataset = None
        if test_files:
            test_dataset, test_data_size = to_textline_dataset(
                test_files[self.task_index], self.config)
            test_steps = test_data_size // self.config["batch_size"] + 1
            self.config["test_steps"] = test_steps
            print("Number of test records for this worker: ", test_data_size)
            print("Number of test steps for this worker: ", self.config["test_steps"])
        return main(train_dataset, test_dataset, self.tf_config, self.server, self.config)

    def setup_address(self):
        ip = self.get_node_ip()
        port = find_free_port()
        self.address = f"{ip}:{port}"
        return self.address

    def get_address(self):
        return self.address

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return ray._private.services.get_node_ip_address()


def to_tensor_slice_dataset(data, label, config):
    # TODO: make OrderedDict more general
    features = collections.OrderedDict()
    output_types = collections.OrderedDict()
    for i in range(len(CONTINUOUS_COLUMNS)):
        import numpy as np
        features[CONTINUOUS_COLUMNS[i]] = data[i].astype(np.float32)
        output_types[CONTINUOUS_COLUMNS[i]] = tf.float32
    for j in range(len(CATEGORICAL_COLUMNS)):
        # EmbeddingVariable only supports int64 not int32.
        features[CATEGORICAL_COLUMNS[j]] = data[j + len(CONTINUOUS_COLUMNS)].astype('int64')
        output_types[CATEGORICAL_COLUMNS[j]] = tf.int64
        labels = label

    def get_item():
        i = 0
        size = len(labels)
        while i < size:
            single_features = collections.OrderedDict()
            for k, v in features.items():
                single_features[k] = v[i]
            single_label = labels[i]
            yield single_features, single_label
            i += 1

    # dataset = tf.data.Dataset.from_generator(get_item, output_types=(output_types, tf.int32))

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=20000,
                              seed=config["seed"])  # fix seed for reproducing
    dataset = dataset.repeat(config["no_of_epochs"])
    dataset = dataset.prefetch(config["batch_size"])
    dataset = dataset.batch(config["batch_size"])
    dataset = dataset.prefetch(2)
    return dataset


def to_textline_dataset(files_with_sizes, config):

    def parse_csv(value):
        cont_defaults = [[0.0] for i in range(1, 14)]
        cate_defaults = [[0] for i in range(1, 27)]
        label_defaults = [[0]]
        column_headers = TRAIN_DATA_COLUMNS
        record_defaults = label_defaults + cont_defaults + cate_defaults
        columns = tf.io.decode_csv(value, record_defaults=record_defaults)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        labels = all_columns.pop(LABEL_COLUMN[0])
        features = all_columns
        # EmbeddingVariable only supports int64 not int32.
        for j in range(len(CATEGORICAL_COLUMNS)):
            features[CATEGORICAL_COLUMNS[j]] = \
                tf.cast(features[CATEGORICAL_COLUMNS[j]], dtype=tf.int64)
        return features, labels

    print(files_with_sizes)
    files = [pair[0] for pair in files_with_sizes]
    data_size = sum([pair[1] for pair in files_with_sizes])

    dataset = tf.data.TextLineDataset(files)
    dataset = dataset.shuffle(buffer_size=20000,
                              seed=config["seed"])  # fix seed for reproducing
    dataset = dataset.repeat(config["no_of_epochs"])
    dataset = dataset.prefetch(config["batch_size"])
    dataset = dataset.batch(config["batch_size"])
    dataset = dataset.map(parse_csv, num_parallel_calls=28)
    dataset = dataset.prefetch(2)
    return dataset, data_size


def data_processing(args):
    print("Checking dataset...")
    train_file = args.data_location + '/train.csv'
    test_file = args.data_location + '/eval.csv'
    if (not os.path.exists(train_file)) or (not os.path.exists(test_file)):
        invalidInputError(False, "Dataset does not exist in the given data_location.")

    train_tbl = FeatureTable.read_csv(train_file, names=TRAIN_DATA_COLUMNS)
    test_tbl = FeatureTable.read_csv(test_file, names=TRAIN_DATA_COLUMNS)

    no_of_training_examples = train_tbl.size()
    no_of_test_examples = test_tbl.size()
    print("The size of the training dataset is {}".format(no_of_training_examples))
    print("The size of the test dataset is {}".format(no_of_test_examples))

    # set batch size, epoch & steps
    batch_size = math.ceil(
        args.batch_size / args.micro_batch
    ) if args.micro_batch and not args.tf else args.batch_size

    if args.steps == 0:
        no_of_epochs = 1
        train_steps = math.ceil(
            (float(no_of_epochs) * no_of_training_examples) / batch_size)
    else:
        no_of_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples)
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples) / batch_size)
    print("The training steps is {}".format(train_steps))
    print("The test steps is {}".format(test_steps))

    # set directory path for checkpoint_dir
    model_dir = os.path.join(args.output_dir,
                             'model_WIDE_AND_DEEP_' + str(int(time.time())))
    checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    print("Saving model checkpoints to " + checkpoint_dir)
    params = {
        'checkpoint_dir': checkpoint_dir,
        'train_steps': train_steps,
        # 'test_steps': test_steps,
        'batch_size': batch_size,
        'no_of_epochs': no_of_epochs
    }

    # Preprocessing
    train_tbl = train_tbl.hash_encode(columns=CATEGORICAL_COLUMNS, bins=10000)
    test_tbl = test_tbl.hash_encode(columns=CATEGORICAL_COLUMNS, bins=10000)
    # Category encode results in a larger vocabulary and will make model larger
    # train_tbl, indices = train_tbl.category_encode(columns=CATEGORICAL_COLUMNS, freq_limit=10)
    # test_tbl = test_tbl.encode_string(columns=CATEGORICAL_COLUMNS, indices=indices)

    train_tbl = train_tbl.fillna(0.0, CONTINUOUS_COLUMNS)
    train_tbl = train_tbl.fillna(0, CATEGORICAL_COLUMNS)
    test_tbl = test_tbl.fillna(0.0, CONTINUOUS_COLUMNS)
    test_tbl = test_tbl.fillna(0, CATEGORICAL_COLUMNS)

    return train_tbl.df, test_tbl.df, params


parser = get_arg_parser()
args = parser.parse_args()
# TODO: add jemalloc
if not args.tf:
    set_env_for_DeepRec()

extra_params = \
    {"min-worker-port": "30000", "max-worker-port": "33333", "metrics-export-port": "20010"}
if args.cluster_mode == "k8s":
    if not args.master:
        invalidInputError(False, "k8s master address must be provided for k8s cluster_mode")
    conf = {
        "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName":
            "nfsvolumeclaim",
        "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path":
            "/bigdl2.0/data",
        "spark.kubernetes.executor.deleteOnTermination": "true",
        "spark.kubernetes.memoryOverheadFactor": "0.6"
    }
    sc = init_orca_context(
        cluster_mode="k8s", cores=args.cores, num_nodes=args.num_nodes, memory="40g",
        master=args.master, container_image="intelanalytics/bigdl-k8s:latest",
        conf=conf, extra_params=extra_params, init_ray_on_spark=True)
elif args.cluster_mode == "yarn":
    sc = init_orca_context(
        cluster_mode="yarn", cores=args.cores, num_nodes=args.num_nodes, memory="40g",
        conf={"spark.yarn.executor.memoryOverhead": "24000"},
        extra_params=extra_params, init_ray_on_spark=True)
elif args.cluster_mode == "local":
    invalidInputError(
        args.instances_per_node >= 2,
        "For local cluster_mode, instances_per_node needs to be no less than 2")
    invalidInputError(args.num_nodes == 1, "For local cluster mode, num_nodes must be equal to 1")
    init_orca_context(cores=args.cores, memory="20g", init_ray_on_spark=True)
else:
    invalidInputError(
        False,
        "cluster_mode should be one of 'local', 'k8s' and 'yarn', but got " + args.cluster_mode)

train_df, test_df, params = data_processing(args)
config = vars(args)
config.update(params)

cluster = RayDeepRecCluster(config, args.instances_per_node, args.num_ps)
cluster.fit(train_df, test_df, args.in_memory, FEATURE_COLUMNS, LABEL_COLUMN)

stop_orca_context()
