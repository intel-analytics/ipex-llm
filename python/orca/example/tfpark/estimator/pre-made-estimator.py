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
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import pandas as pd
import tensorflow as tf
from optparse import OptionParser
from tensorflow_estimator.python.estimator.canned import prediction_keys
from zoo import init_nncontext
from zoo.tfpark import TFDataset, TFEstimator
from zoo.tfpark import ZooOptimizer


def make_input_fn(data_df, label_df, mode, batch_size=-1, batch_per_thread=-1):
    if mode == tf.estimator.ModeKeys.TRAIN:
        def input_function():
            ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
            ds = ds.shuffle(1000)
            ds = TFDataset.from_tf_data_dataset(dataset=ds, batch_size=batch_size)
            return ds
    elif mode == tf.estimator.ModeKeys.EVAL:
        def input_function():
            ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
            ds = TFDataset.from_tf_data_dataset(dataset=ds, batch_per_thread=batch_per_thread)
            return ds
    else:
        def input_function():
            ds = tf.data.Dataset.from_tensor_slices((dict(data_df),))
            ds = TFDataset.from_tf_data_dataset(dataset=ds, batch_size=batch_size,
                                                batch_per_thread=batch_per_thread)
            return ds
    return input_function


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--data_dir", dest="data_dir")
    (options, args) = parser.parse_args(sys.argv)

    dftrain = pd.read_csv(os.path.join(options.data_dir, 'train.csv'))
    dfeval = pd.read_csv(os.path.join(options.data_dir, 'eval.csv'))
    y_train = dftrain.pop('survived')
    y_eval = dfeval.pop('survived')

    CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                           'embark_town', 'alone']
    NUMERIC_COLUMNS = ['age', 'fare']

    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        vocabulary = dftrain[feature_name].unique()
        feature_columns.append(tf.feature_column.
                               categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    sc = init_nncontext()

    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                               optimizer=ZooOptimizer(tf.train.FtrlOptimizer(0.2)),
                                               model_dir="/tmp/estimator/linear")
    zoo_est = TFEstimator(linear_est)
    train_input_fn = make_input_fn(dftrain, y_train,
                                   mode=tf.estimator.ModeKeys.TRAIN,
                                   batch_size=32)
    zoo_est.train(train_input_fn, steps=200)

    eval_input_fn = make_input_fn(dfeval, y_eval,
                                  mode=tf.estimator.ModeKeys.EVAL,
                                  batch_per_thread=8)
    eval_result = zoo_est.evaluate(eval_input_fn, ["acc"])
    print(eval_result)

    pred_input_fn = make_input_fn(dfeval, y_eval,
                                  mode=tf.estimator.ModeKeys.PREDICT,
                                  batch_per_thread=8)
    predictions = zoo_est.predict(pred_input_fn,
                                  predict_keys=[prediction_keys.PredictionKeys.CLASS_IDS])
    print(predictions.collect())
    print("finished...")
    sc.stop()
