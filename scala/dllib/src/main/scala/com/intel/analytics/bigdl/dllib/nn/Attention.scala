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
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.language.existentials
import scala.reflect.ClassTag

/**
 * Implementation of multiheaded attention and self-attention layers.
 *
 * @param hiddenSize hidden size
 * @param numHeads heads number
 * @param attentionDropout
 */
class Attention[T: ClassTag](val hiddenSize: Int, val numHeads: Int, val attentionDropout: Float)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Activity, T] {

  // for prediction
  private val joinK = nn.JoinTable[T](dimension = 2, nInputDims = -1)
  private val joinV = nn.JoinTable[T](dimension = 2, nInputDims = -1)

  private val queryLayer = TransformerOperation.dense(
    hiddenSize, hiddenSize, false, name = s"${this.getName()}_q")
  private val keyLayer = TransformerOperation.dense(
    hiddenSize, hiddenSize, false, name = s"${this.getName()}_k")
  private val valueLayer = TransformerOperation.dense(
    hiddenSize, hiddenSize, false, name = s"${this.getName()}_v")

  private val querySplitLayer = new SplitHeads(hiddenSize, numHeads, true)
  private val keySplitLayer = new SplitHeads(hiddenSize, numHeads)
  private val valueSplitLayer = new SplitHeads(hiddenSize, numHeads)

  private val contiguousQLayer = new Contiguous[T]()
  private val contiguousKLayer = new Contiguous[T]()
  private val contiguousVLayer = new Contiguous[T]()
  private val matmulLayer = MM(transB = true)
  private val caddLayer = CAddTable()
  private val softMaxLayer = TransformerOperation.softMax[T]()
  private val dropLayer = Dropout(initP = (1.0 - attentionDropout))
  private val matmulNoTransLayer = MM()
  // Recombine heads --> (batch_size, length, hidden_size)
  private val combineHeadsLayer = new CombineHeads()
  // Run the combined outputs through another linear projection layer.
  private val outputLayer = TransformerOperation.dense(
    hiddenSize, hiddenSize, false, name = s"${this.getName()}_output_transform")

  private[bigdl] val model : Module[T] = {
    // InputX with shape (batch_size, length_x, hidden_size).
    // InputY with shape (batch_size, length_x, hidden_size)
    // for self attention, InputX and InputY should be the same.
    // Bias is attention bias that will be added to the result of the dot product.
    val inputX = Input()
    val inputY = Input()
    val inputBias = Input()

    val queryNode = queryLayer.inputs(inputX)
    val keyNode = keyLayer.inputs(inputY)
    val valueNode = valueLayer.inputs(inputY)

    val model = Graph(Array(inputX, inputY, inputBias),
      Array(createModule(queryNode, keyNode, valueNode, inputBias)))
    if (this.train) model.training() else model.evaluate()
  }

  private val graph: Module[T] = {
    val queryNode = Input()
    val keyNode = Input()
    val valueNode = Input()
    val inputBias = Input()

    Graph(Array(queryNode, keyNode, valueNode, inputBias),
      Array(createModule(queryNode, keyNode, valueNode, inputBias)))
  }

  private def createModule(inputQuery: ModuleNode[T], inputKey: ModuleNode[T],
    inputValue: ModuleNode[T], inputBias: ModuleNode[T]) : ModuleNode[T] = {
    val querySplit = querySplitLayer.inputs(inputQuery)
    val keySplit = keySplitLayer.inputs(inputKey)
    val valueSplit = valueSplitLayer.inputs(inputValue)

    val contiguousQ = contiguousQLayer.inputs(querySplit)
    val contiguousK = contiguousKLayer.inputs(keySplit)
    val contiguousV = contiguousVLayer.inputs(valueSplit)

    val matmul = matmulLayer.inputs(contiguousQ, contiguousK)
    val cadd = caddLayer.inputs(matmul, inputBias)
    val softMax = softMaxLayer.inputs(cadd)

    val drop = dropLayer.inputs(softMax)
    val matmulNoTrans = matmulNoTransLayer.inputs(drop, contiguousV)
    // Recombine heads --> (batch_size, length, hidden_size)
    val combineHeads = combineHeadsLayer.inputs(matmulNoTrans)
    outputLayer.inputs(combineHeads)
  }


  private def updateOutputCache(input: Activity): Activity = {
    require(!this.isTraining(), "Only support input cache for model inference")
    val inputTable = input.toTable
    val inputX = inputTable[Tensor[T]](1)
    val inputY = inputTable[Tensor[T]](2)
    val inputBias = inputTable[Table](3).apply[Tensor[T]](1)
    /**
     * cache: (Used during prediction) dictionary with tensors containing results of
     * previous attentions. The dictionary must have the items:
     * {"k": tensor with shape [batch_size, i, key_channels],
     * "v": tensor with shape [batch_size, i, value_channels]}
     * where i is the current decoded length.
     */
    val cache = inputTable[Table](3).apply[Table](2)

    val query = queryLayer.forward(inputX).toTensor[T]

    val (inputK, inputV) = if (cache.length() > 0) {
      (cache.apply[Tensor[T]](this.getName() + "_k"),
        cache.apply[Tensor[T]](this.getName() + "_v"))
    } else (null, null)

    val key = if (inputK != null && !inputK.isEmpty) {
      joinK.forward(T(keyLayer.forward(inputY).toTensor[T], inputK))
    } else keyLayer.forward(inputY).toTensor[T]
    val value = if (inputV != null && !inputV.isEmpty) {
      joinV.forward(T(valueLayer.forward(inputY).toTensor[T], inputV))
    } else valueLayer.forward(inputY).toTensor[T]

    // update cache
    if (cache.length() > 0) {
      cache.update(this.getName() + "_k", key)
      cache.update(this.getName() + "_v", value)
    }
    output = graph.updateOutput(T(query, key, value, inputBias))
    output
  }
  override def updateOutput(input: Activity): Activity = {
    require(input.toTable.length() == 3,
      s"only support 3 inputs, but get ${input.toTable.length()}")

    val cache = input.toTable.apply[Activity](3)
    if (cache.isInstanceOf[Tensor[T]]) {
      output = model.updateOutput(input)
    } else if (cache.isInstanceOf[Table]) {
      output = updateOutputCache(input)
    }
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = model.updateGradInput(input, gradOutput)
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    model.accGradParameters(input, gradOutput)
  }

  override def training(): this.type = {
    train = true
    model.training()
    this
  }

  override def evaluate(): this.type = {
    train = false
    model.evaluate()
    this
  }

  override def getExtraParameter(): Array[Tensor[T]] = {
    model.getExtraParameter()
  }

  override def getTimes(): Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    model.getTimes()
  }

  override def resetTimes(): Unit = {
    model.resetTimes()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    model.parameters()
  }

  override def getParametersTable(): Table = {
    model.getParametersTable()
  }

  override def clearState(): this.type = {
    model.clearState()
    this
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
