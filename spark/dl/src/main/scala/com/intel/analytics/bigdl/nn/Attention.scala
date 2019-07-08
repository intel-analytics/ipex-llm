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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Implementation of multiheaded attention and self-attention layers.
 * @param hiddenSize hidden size
 * @param numHeads heads number
 * @param attentionDropout
 */
class Attention[T: ClassTag](
  val hiddenSize: Int, val numHeads: Int, val attentionDropout: Float)
 (implicit ev: TensorNumeric[T]) extends BaseModule[T] {

  override def buildModel(): Module[T] = {
    // InputX with shape (batch_size, length_x, hidden_size).
    // InputY with shape (batch_size, length_x, hidden_size)
    // for self attention, InputX and InputY should be the same.
    // Bias is attention bias that will be added to the result of the dot product.
    val inputX = Input()
    val inputY = Input()
    val inputBias = Input()

    // Layers for linearly projecting the queries, keys, and values.
    val queryLayer = TransformerOperation.dense(
      hiddenSize, hiddenSize, false, name = s"${this.getName()}_q").inputs(inputX)
    val keyLayer = TransformerOperation.dense(
      hiddenSize, hiddenSize, false, name = s"${this.getName()}_k").inputs(inputY)
    val valueLayer = TransformerOperation.dense(
      hiddenSize, hiddenSize, false, name = s"${this.getName()}_v").inputs(inputY)

    val querySplit = new SplitHeads(hiddenSize, numHeads, true).inputs(queryLayer)
    val keySplit = new SplitHeads(hiddenSize, numHeads).inputs(keyLayer)
    val valueSplit = new SplitHeads(hiddenSize, numHeads).inputs(valueLayer)

    val contiguousQ = new Contiguous[T]().inputs(querySplit)
    val contiguousK = new Contiguous[T]().inputs(keySplit)
    val contiguousV = new Contiguous[T]().inputs(valueSplit)

    val matmul = MM(transB = true).inputs(contiguousQ, contiguousK)
    val cadd = CAddTable().inputs(matmul, inputBias)
    val softMax = TransformerOperation.softMax[T]().inputs(cadd)

    val drop = Dropout(initP = (1.0 - attentionDropout)).inputs(softMax)
    val matmulNoTrans = MM().inputs(drop, contiguousV)
    // Recombine heads --> (batch_size, length, hidden_size)
    val combineHeads = new CombineHeads().inputs(matmulNoTrans)
    // Run the combined outputs through another linear projection layer.
    val outputLayer = TransformerOperation.dense(
      hiddenSize, hiddenSize, false, name = s"${this.getName()}_output_transform")
      .inputs(combineHeads)
    val graph = Graph(Array(inputX, inputY, inputBias), Array(outputLayer))
    graph
  }
}
// Combine tensor that has been splitted.
//  input should be tensor with shape (batch_size, num_heads, length, hidden_size/num_heads)
// output should be tensor with shape (batch_size, length, hidden_size)
private[nn] class CombineHeads[T: ClassTag](implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  private val permutations: (Int, Int) = (2, 3)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val batchSize = input.size(1)
    val length = input.size(3)
    val hiddenSize = input.size(2) * input.size(4)

    output.resizeAs(input).copy(input)
    output = output.transpose(permutations._1, permutations._2)
      .reshape(Array(batchSize, length, hiddenSize))
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val size = Array(input.size(1), input.size(3), input.size(2), input.size(4))
    if (gradOutput.isContiguous()) {
      gradInput = gradOutput.view(size)
    } else {
      gradInput = gradOutput.contiguous().view(size)
    }
    gradInput = gradInput.transpose(permutations._1, permutations._2).contiguous()
    gradInput
  }
}

/**
 * Split x into different heads, and transpose the resulting value.
 * The tensor is transposed to insure the inner dimensions hold the correct
 * values during the matrix multiplication.
 * input with shape (batch_size, length, hidden_size)
 * output with shape (batch_size, num_heads, length, hidden_size/num_heads)
 * @param hiddenSize
 * @param numHeads
 * @param mul
 * @tparam T The numeric type in this module parameters
 */
private[nn] class SplitHeads[T: ClassTag](val hiddenSize: Int, val numHeads: Int,
  val mul: Boolean = false)(implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  private val depth = hiddenSize / numHeads
  private val value = ev.fromType(math.pow(depth, -0.5))
  private val permutations: (Int, Int) = (2, 3)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val batchSize = input.size(1)
    val length = input.size(2)

    output.resizeAs(input).copy(input)
    output = output.reshape(Array(batchSize, length, numHeads, depth))
      .transpose(permutations._1, permutations._2)
    if (mul) {
      output.mul(value)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (mul) {
      gradInput.resizeAs(gradOutput).zero().add(value, gradOutput)
    } else {
      gradInput.resizeAs(gradOutput).copy(gradOutput)
    }
    gradInput = gradInput.transpose(permutations._1, permutations._2).contiguous()
    gradInput.resize(input.size())
    gradInput
  }
}

object Attention {
  def apply[@specialized(Float, Double) T: ClassTag]
  (hiddenSize: Int, numHeads: Int, attentionDropout: Float)
  (implicit ev: TensorNumeric[T]): Attention[T] =
    new Attention(hiddenSize: Int, numHeads: Int, attentionDropout: Float)
}
