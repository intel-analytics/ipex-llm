/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.nn.rnn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{Sequential, Module => _, _}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.Dense
import com.intel.analytics.bigdl.nn.ops.Gather
import com.intel.analytics.bigdl.nn.tf.TensorArrayScatter
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T

import scala.reflect.ClassTag

/**
 * Implementation of fully connected network.
 * Input with shape [batch_size, length, hidden_size]
 * Output with shape [batch_size, length, hidden_size]
 * @param hiddenSize hidden_size
 * @param filterSize
 * @param reluDropout
 */
private[nn] class FeedForwardNetwork[T: ClassTag](hiddenSize: Int, filterSize: Int,
                                      reluDropout: Float)(implicit ev: TensorNumeric[T])
  extends BaseModule[T]{

  override def buildModel(): Module[T] = {
    val input = Input()
    val filterLayer = TransformerOperation.dense(
      hiddenSize, filterSize, bias = true, activation = ReLU[T]()).inputs(input)
    val drop = if (train) {
      Dropout(initP = (1.0 - reluDropout)).inputs(filterLayer)
    } else filterLayer
    val output_dense_layer = TransformerOperation.dense(
      filterSize, hiddenSize, bias = true).inputs(drop)
    val graph = Graph(Array(input), Array(output_dense_layer))
    if (this.train) graph.training() else graph.evaluate()
    graph
  }
}
