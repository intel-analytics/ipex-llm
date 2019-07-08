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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{Module => _}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Implementation FeedForwardNetwork constructed with fully connected network.
 * Input with shape (batch_size, length, hidden_size)
 * Output with shape (batch_size, length, hidden_size)
 * @param hiddenSize hidden_size
 * @param filterSize
 * @param reluDropout
 */
class FeedForwardNetwork[T: ClassTag](val hiddenSize: Int, val filterSize: Int,
                                      val reluDropout: Float)(implicit ev: TensorNumeric[T])
  extends BaseModule[T]{

  override def buildModel(): Module[T] = {
    val input = Input()
    val filterLayer = TransformerOperation.dense(
      hiddenSize, filterSize, bias = true, activation = ReLU[T](),
      name = s"${this.getName()}_filter_layer").inputs(input)
    val drop = Dropout(initP = (1.0 - reluDropout)).inputs(filterLayer)
    val output_dense_layer = TransformerOperation.dense(
      filterSize, hiddenSize, bias = true, name = s"${this.getName()}_output_layer").inputs(drop)
    val graph = Graph(Array(input), Array(output_dense_layer))
    graph
  }
}

object FeedForwardNetwork {
  def apply[@specialized(Float, Double) T: ClassTag](
    hiddenSize: Int,
    filterSize: Int,
    reluDropout: Float)
  (implicit ev: TensorNumeric[T]): FeedForwardNetwork[T] =
    new FeedForwardNetwork[T](hiddenSize, filterSize, reluDropout)
}
