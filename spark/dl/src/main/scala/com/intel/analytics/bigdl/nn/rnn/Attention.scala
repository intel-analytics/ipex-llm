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
 * @param hiddenSize hidden size
 * @param numHeads heads number
 * @param attentionDropout
 */
private[nn] class AttentionLayer[T: ClassTag](
  hiddenSize: Int, numHeads: Int, attentionDropout: Float)
 (implicit ev: TensorNumeric[T]) extends BaseModule[T] {

  override def buildModel(): Module[T] = {
    // InputX with shape [batch_size, length_x, hidden_size].
    // InputY with shape [batch_size, length_x, hidden_size]
    // for self attention, InputX and InputY should be the same.
    // Bias is attention bias that will be added to the result of the dot product.
    val inputX = Input()
    val inputY = Input()
    val inputBias = Input()

    // Layers for linearly projecting the queries, keys, and values.
    val queryLayer = TransformerOperation.dense(
      hiddenSize, hiddenSize, false, name = "q").inputs(inputX)
    val keyLayer = TransformerOperation.dense(
      hiddenSize, hiddenSize, false, name = "k").inputs(inputY)
    val valueLayer = TransformerOperation.dense(
      hiddenSize, hiddenSize, false, name = "v").inputs(inputY)

    val querySplit = new SplitHeads(hiddenSize, numHeads, true).inputs(queryLayer)
    val keySplit = new SplitHeads(hiddenSize, numHeads).inputs(keyLayer)
    val valueSplit = new SplitHeads(hiddenSize, numHeads).inputs(valueLayer)

    val contiguousQ = new Contiguous[T]().inputs(querySplit)
    val contiguousK = new Contiguous[T]().inputs(keySplit)
    val contiguousV = new Contiguous[T]().inputs(valueSplit)

    val matmul = MM(transB = true).inputs(contiguousQ, contiguousK)
    val cadd = CAddTable().inputs(matmul, inputBias)
    val softMax = TransformerOperation.softMax[T]().inputs(cadd)

    val drop = if (train) {
      Dropout(initP = (1.0 - attentionDropout)).inputs(softMax)
    } else softMax
    val matmulNoTrans = MM().inputs(drop, contiguousV)
    // Recombine heads --> [batch_size, length, hidden_size]
    val combineHeads = new CombineHeads().inputs(matmulNoTrans)
    // Run the combined outputs through another linear projection layer.
    val outputLayer = TransformerOperation.dense(
      hiddenSize, hiddenSize, false, name = "output_transform").inputs(combineHeads)
    val graph = Graph(Array(inputX, inputY, inputBias), Array(outputLayer))
    if (this.train) graph.training() else graph.evaluate()
    graph
  }
}

object Attention {
  def apply[@specialized(Float, Double) T: ClassTag]
  (hiddenSize: Int, numHeads: Int, attentionDropout: Float)
  (implicit ev: TensorNumeric[T]): AttentionLayer[T] =
    new AttentionLayer(hiddenSize: Int, numHeads: Int, attentionDropout: Float)
}