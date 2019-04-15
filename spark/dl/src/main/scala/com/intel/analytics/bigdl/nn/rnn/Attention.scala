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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.Dense
import com.intel.analytics.bigdl.nn.ops.BatchMatMul
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.bigdl.nn.keras

import scala.reflect.ClassTag

class AttentionLayer[T: ClassTag](hidden_size: Int, num_heads: Int, attention_dropout: Float)
                                 (implicit ev: TensorNumeric[T]) extends BaseModule[T] {

  override var model : Module[T] = buildModel()

  private def buildModel(): Module[T] = {
    val inputX = Input()
    val inputY = Input()
    val inputBias = Input()

    val q_dense_layer = new KerasWrapper(
      new Dense(outputDim = hidden_size, bias = false)).inputs(inputX)
    val k_dense_layer = new KerasWrapper(
      new Dense(outputDim = hidden_size, bias = false)).inputs(inputY)
    val v_dense_layer = new KerasWrapper(
      new Dense(outputDim = hidden_size, bias = false)).inputs(inputY)

    val split_q = new SplitHeads(hidden_size, num_heads, true).inputs(q_dense_layer)
    val split_k = new SplitHeads(hidden_size, num_heads).inputs(k_dense_layer)
    val split_v = new SplitHeads(hidden_size, num_heads).inputs(v_dense_layer)

    val contiguous_q = new Contiguous[T]().inputs(split_q)
    val contiguous_k = new Contiguous[T]().inputs(split_k)
    val contiguous_v = new Contiguous[T]().inputs(split_v)

    val matmul = MM(transB = true).inputs(contiguous_q, contiguous_k)
    val cadd = CAddTable().inputs(matmul, inputBias)
    val softMax = new KerasWrapper(keras.SoftMax()).inputs(cadd)
    val drop = Dropout(initP = (1.0 - attention_dropout)).inputs(softMax)

    val matmulNoTrans = MM().inputs(drop, contiguous_v)
    val combineHeads = new CombineHeads().inputs(matmulNoTrans)
    val output_dense_layer = new KerasWrapper(new Dense(outputDim = hidden_size, bias = false)
    ).inputs(combineHeads)
    val graph = Graph(Array(inputX, inputY, inputBias), Array(output_dense_layer))
    if (this.train) graph.training() else graph.evaluate()
    graph
  }
}

object SelfAttention {
  def apply[@specialized(Float, Double) T: ClassTag]
  (hidden_size: Int, num_heads: Int, attention_dropout: Float)
  (implicit ev: TensorNumeric[T]): AttentionLayer[T] =
    new AttentionLayer(hidden_size: Int, num_heads: Int, attention_dropout: Float)
}