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
import com.intel.analytics.bigdl.nn.{Sequential, keras, _}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, T}

import scala.reflect.ClassTag

/**
 * Implementation of multiheaded attention and self-attention layers.
 * @param hidden_size hidden size
 * @param num_heads heads number
 * @param attention_dropout
 */
private[nn] class AttentionLayer[T: ClassTag](
  hidden_size: Int, num_heads: Int, attention_dropout: Float)
 (implicit ev: TensorNumeric[T]) extends BaseModule[T] {

  override var model : Module[T] = buildModel()

  private def buildModel(): Module[T] = {
    // InputX with shape [batch_size, length_x, hidden_size].
    // InputY with shape [batch_size, length_x, hidden_size]
    // for self attention, InputX and InputY should be same.
    // Bias is attention bias that will be added to the result of the dot product.
    val inputX = Input()
    val inputY = Input()
    val inputBias = Input()

    val q_dense_layer = TransformerOperation.dense(
      hidden_size, hidden_size, false, name = "q").inputs(inputX)
    val k_dense_layer = TransformerOperation.dense(
      hidden_size, hidden_size, false, name = "k").inputs(inputY)
    val v_dense_layer = TransformerOperation.dense(
      hidden_size, hidden_size, false, name = "v").inputs(inputY)

    val split_q = new SplitHeads(hidden_size, num_heads, true).inputs(q_dense_layer)
    val split_k = new SplitHeads(hidden_size, num_heads).inputs(k_dense_layer)
    val split_v = new SplitHeads(hidden_size, num_heads).inputs(v_dense_layer)

    val contiguous_q = new Contiguous[T]().inputs(split_q)
    val contiguous_k = new Contiguous[T]().inputs(split_k)
    val contiguous_v = new Contiguous[T]().inputs(split_v)

    val matmul = MM(transB = true).inputs(contiguous_q, contiguous_k)
    val cadd = CAddTable().inputs(matmul, inputBias)
    val softMax = TransformerOperation.softMax[T]().inputs(cadd)

    val drop = if (train) {
      Dropout(initP = (1.0 - attention_dropout)).inputs(softMax)
    } else softMax
    val matmulNoTrans = MM().inputs(drop, contiguous_v)
    // Recombine heads --> [batch_size, length, hidden_size]
    val combineHeads = new CombineHeads().inputs(matmulNoTrans)
    // Run the combined outputs through another linear projection layer.
    val output_dense_layer = TransformerOperation.dense(
      hidden_size, hidden_size, false, name = "output_transform").inputs(combineHeads)
    val graph = Graph(Array(inputX, inputY, inputBias), Array(output_dense_layer))
    if (this.train) graph.training() else graph.evaluate()
    graph
  }
}

object Attention {
  def apply[@specialized(Float, Double) T: ClassTag]
  (hidden_size: Int, num_heads: Int, attention_dropout: Float)
  (implicit ev: TensorNumeric[T]): AttentionLayer[T] =
    new AttentionLayer(hidden_size: Int, num_heads: Int, attention_dropout: Float)
}