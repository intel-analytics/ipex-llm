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

# script to generate multi_type_inputs_outputs.pb
import tensorflow as tf

from tensorflow.python.platform import gfile

float_input = tf.placeholder(dtype=tf.float32, name="float_input", shape=(None, 1))
double_input = tf.placeholder(dtype=tf.float64, name="double_input", shape=(None, 1))
int_input = tf.placeholder(dtype=tf.int32, name="int_input", shape=(None, 1))
long_input = tf.placeholder(dtype=tf.int64, name="long_input", shape=(None, 1))
uint8_input = tf.placeholder(dtype=tf.uint8, name="uint8_input", shape=(None, 1))

float_output = tf.identity(float_input, name="float_output")
double_output = tf.identity(double_input, name="double_output")
int_output = tf.identity(int_input, name="int_output")
long_output = tf.identity(long_input, name="long_output")
uint8_output = tf.identity(uint8_input, name="uint8_output")

with gfile.GFile("./multi_type_inputs_outputs.pb", "wb") as f:
    f.write(tf.get_default_graph().as_graph_def().SerializeToString())