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

# This would makes sure Python is aware there is more than one sub-package within bigdl,
# physically located elsewhere.
# Otherwise there would be module not found error in non-pip's setting as Python would
# only search the first bigdl package and end up finding only one sub-package.

"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import absl
import sys
import modeling
import optimization
import mlp_logging as mllog
from mlperf_logging.mllog import constants as mllog_constants

import tensorflow.compat.v1 as tf
# from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
# from tensorflow.contrib import data as contrib_data
# from tensorflow.contrib import tpu as contrib_tpu
from bigdl.orca.learn.tf.tf_estimator import TFEstimator

flags = absl.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_enum("optimizer", "adamw", ["adamw", "lamb"],
                  "The optimizer for training.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("poly_power", 1.0, "The power of poly decay.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("start_warmup_step", 0, "The starting step of warmup.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

# flags.DEFINE_integer(
#     "num_gpus", 0,
#     "Use the GPU backend if this value is set to more than zero.")

flags.DEFINE_integer("steps_per_update", 1,
                     "The number of steps for accumulating gradients.")

flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "The maximum number of checkpoints to keep.")

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings, optimizer, poly_power,
                     start_warmup_step, steps_per_update,
                     num_workers):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    tf.logging.info(f"********mode in model_fn*******: {mode}")

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    global_batch_size = params["batch_size"]
    tf.logging.info(f"*** Global batch size in model_fn is {global_batch_size}")
    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights,
         num_workers)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels, global_batch_size)

    total_loss = masked_lm_loss + next_sentence_loss

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

      tvar_index = {var.name.replace(":0", ""): var for var in tvars}
      assignment_map = collections.OrderedDict([
          (name, tvar_index.get(name, value))
          for name, value in assignment_map.items()
      ])

      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, False,
          optimizer, poly_power, start_warmup_step, steps_per_update)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      ])
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=metric_fn(
            masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
            masked_lm_weights, next_sentence_example_loss,
            next_sentence_log_probs, next_sentence_labels))

    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights, num_workers):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator 
    scaled_loss = loss / num_workers

  return (scaled_loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels, global_batch_size):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

    # loss = tf.reduce_mean(per_example_loss)
    scaled_per_example_loss = per_example_loss * (1. / global_batch_size)
    scaled_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

    # print_op = tf.print("get_next_sentence_output", scaled_loss, output_stream=sys.stdout)
    # with tf.control_dependencies([print_op]):
    #   scaled_loss = scaled_loss + 0.0

    return (scaled_loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(input_files,
                     batch_size,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     input_context = None,
                     num_cpu_threads=4,
                     num_eval_steps=1):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  # print(f"*******batch_size in input_fn_builder is {batch_size}")

  def input_fn(params, input_context = None):
    """The actual input function."""

    # batch_size = params["batch_size"]
    print(f"*******batch_size in input_fn is {batch_size}")

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      if input_context:
        tf.logging.info(
            'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
            input_context.input_pipeline_id, input_context.num_input_pipelines))
        d = d.shard(input_context.num_input_pipelines,
                    input_context.input_pipeline_id)
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.data.experimental.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=1000)
      d = d.repeat()
    else:
      d = tf.data.TFRecordDataset(input_files)
      d = d.take(batch_size * num_eval_steps)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


class CheckpointHook(tf.train.CheckpointSaverHook):
  """Add MLPerf logging to checkpoint saving."""

  def __init__(self, num_train_steps, *args, **kwargs):
    super(CheckpointHook, self).__init__(*args, **kwargs)
    self.num_train_steps = num_train_steps
    self.previous_step = None

  def _save(self, session, step):
    if self.previous_step:
      mllog.mllog_end(key=mllog_constants.BLOCK_STOP,
                      metadata={"first_step_num": self.previous_step + 1,
                          "step_count": step - self.previous_step})
    self.previous_step = step
    mllog.mllog_start(key="checkpoint_start", metadata={"step_num" : step}) 
    return_value = super(CheckpointHook, self)._save(session, step)
    mllog.mllog_end(key="checkpoint_stop", metadata={"step_num" : step})
    if step < self.num_train_steps:
        mllog.mllog_start(key=mllog_constants.BLOCK_START,
                          metadata={"first_step_num": step + 1})
    return return_value


def run(_):
  from bigdl.orca import init_orca_context
  sc = init_orca_context()
  num_workers = 2
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_one_hot_embeddings=False,
      optimizer=FLAGS.optimizer,
      poly_power=FLAGS.poly_power,
      start_warmup_step=FLAGS.start_warmup_step,
      steps_per_update=FLAGS.steps_per_update,
      num_workers=num_workers)
  
  train_batch_size_per_worker = int(FLAGS.train_batch_size / num_workers)
  train_input_fn = input_fn_builder(
      input_files=input_files,
      batch_size=train_batch_size_per_worker,
      max_seq_length=FLAGS.max_seq_length,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      is_training=True,
      input_context=None,
      num_cpu_threads=8)

  eval_batch_size_per_worker = int(FLAGS.eval_batch_size / num_workers)
  eval_input_fn = input_fn_builder(
        input_files=input_files,
        batch_size=eval_batch_size_per_worker,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False,
        input_context=None,
        num_cpu_threads=8,
        num_eval_steps=FLAGS.max_eval_steps)
  
  flag_dict = dict(
    output_dir = FLAGS.output_dir,
    keep_checkpoint_max = FLAGS.keep_checkpoint_max,
    save_checkpoints_steps = FLAGS.save_checkpoints_steps,
    num_train_steps = FLAGS.num_train_steps,
    train_batch_size = FLAGS.train_batch_size,
    max_eval_steps = FLAGS.max_eval_steps,
  )
  
  hparams = {"batch_size": flag_dict["train_batch_size"]}

  config = dict(
    inter_op_parallelism=8,
    log_step_count_steps=5,
    keep_checkpoint_max=flag_dict["keep_checkpoint_max"],
    save_checkpoints_steps=flag_dict["save_checkpoints_steps"],
  )
  estimator = TFEstimator(model_fn=model_fn,
                          model_dir=flag_dict["output_dir"],
                          config=config,
                          params=hparams,
                          workers_per_node=num_workers,
                          )

  checkpoint_hook = CheckpointHook(
    num_train_steps=flag_dict["num_train_steps"],
    checkpoint_dir=flag_dict["output_dir"],
    save_steps=flag_dict["save_checkpoints_steps"])
  train_spec=tf.estimator.TrainSpec(input_fn=train_input_fn, 
                                    max_steps=flag_dict["num_train_steps"],
                                    hooks=[checkpoint_hook]
                                    )
  eval_spec=tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                  steps=flag_dict["max_eval_steps"])
  mllog.mlperf_submission_log()
  mllog.mlperf_run_param_log()
  mllog.mllog_end(key=mllog_constants.INIT_STOP)
  mllog.mllog_start(key=mllog_constants.RUN_START)
  results = estimator.train_and_evaluate(train_spec, eval_spec)
  print(results)
  mllog.mllog_end(key=mllog_constants.RUN_STOP)
  # output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
  # with tf.gfile.GFile(output_eval_file, "w") as writer:
  #   tf.logging.info("***** Eval results *****")
  #   for key in sorted(result.keys()):
  #     tf.logging.info("  %s = %s", key, str(result[key]))
  #     writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  absl.app.run(run)
