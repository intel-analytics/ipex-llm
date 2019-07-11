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

import breeze.linalg.*
import breeze.numerics.exp
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{EngineType, T}

import scala.reflect.ClassTag

private[nn] object TransformerOperation {
  def dense[T: ClassTag](
    inputSize: Int,
    outputSize: Int,
    bias: Boolean = true,
    activation: TensorModule[T] = null,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    name: String = "")(implicit ev: TensorNumeric[T]): Module[T] = {
    val seq = new Sequential[T]()
    val layer = Linear[T](
      inputSize = inputSize,
      outputSize = outputSize,
      withBias = bias,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer)

    layer.setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
    if (name != "") layer.setName(name)
    seq.add(TimeDistributed[T](layer))
    if (activation != null) seq.add(activation)
    seq
  }

  def softMax[T: ClassTag]()(implicit ev: TensorNumeric[T]): Module[T] = {
    val layer = SoftMax[T]()
    val model = Sequential[T]()
    model.add(Transpose[T](Array((2, 4))))
    model.add(layer)
    model.add(Transpose[T](Array((2, 4))))
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

  /**
   * Calculate bias tensor from padding values in tensor.
   * Bias tensor that is added to the pre-softmax multi-headed attention logits,
   * which has shape [batch_size, num_heads, length, length]. The tensor is zero at
   * non-padding locations, and -1e9 (negative infinity) at padding locations.
   * Args: x: int tensor with shape [batch_size, length]
   * Returns: Attention bias tensor of shape [batch_size, 1, 1, length].
   * @param input
   * @tparam T
   * @return
   */
  def getPaddingBias[T: ClassTag](input: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val res = getPadding[T](input).mul(ev.fromType(-1e9))
    res.addSingletonDimension(res, 2)
    res.addSingletonDimension(res, 3)
  }

  /**
   * Return float tensor representing the padding values in x.
   * Args:
   * x: int tensor with any shape
   * padding_value: int value that
   * Returns:float tensor with same shape as x containing values 0 or 1.
   *   0 -> non-padding, 1 -> padding
   */
  def getPadding[T: ClassTag](input: Tensor[T], paddingValue: Float = 0.0f)
                             (implicit ev: TensorNumeric[T]): Tensor[T] = {
    input.apply1(e => {if (e == paddingValue) ev.one else ev.zero})
  }

  // Shift the second dimension of x right by one.
  def shiftRight3D[T: ClassTag](input: Tensor[T], output: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    output.resizeAs(input).zero()
    val index = input.size(2)
    output.narrow(2, 2, index - 1).copy(input.narrow(2, 1, index - 1))
    output
  }

  def initRangeTensor[T: ClassTag](length: Int, rangeBuffer: Tensor[T])
    (implicit ev: TensorNumeric[T]): Unit = {
    rangeBuffer.resize(Array(length))
    val arr = rangeBuffer.storage().array()
    for (i <- 0 to (length - 1)) {
      arr(i) = ev.fromType(i)
    }
  }

  /**
   * Args:length: Sequence length.
   * channels: Size of the hidden
   * minTimescale: Minimum scale that will be applied at each position
   * maxTimescale: Maximum scale that will be applied at each position
   * Returns: Tensor with shape [length, hidden_size]
   */
  def getPositionEncode[T: ClassTag](
    length: Int,
    channels: Int,
    minTimescale : Float = 1.0f,
    maxTimescale: Float = 1.0e4f,
    rangeBuffer: Tensor[T],
    outBuffer: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    // get_timing_signal_1d, return (1, length, channels)
    val numTimescales = channels / 2
    val logTimescale = math.log(maxTimescale / minTimescale) /
      math.max(numTimescales - 1, 1)
    // tf.range(num_timescales)
    val invTensor = Tensor[T](1, numTimescales)
    val inv_timescales = invTensor.storage().array()
    val offset = invTensor.storageOffset() - 1
    var i = 0
    while (i < numTimescales) {
      inv_timescales(i + offset) = ev.fromType(minTimescale * math.exp(i * - logTimescale))
      i += 1
    }

    val outSin = outBuffer.narrow(2, 1, numTimescales)
    outSin.addmm(ev.zero, ev.one, rangeBuffer.resize(length, 1), invTensor)
    val outCos = outBuffer.narrow(2, numTimescales + 1, numTimescales).copy(outSin)
    outSin.apply1(e => ev.fromType(math.sin(ev.toType[Float](e))))
    outCos.apply1(e => ev.fromType(math.cos(ev.toType[Float](e))))

    outBuffer
  }

  private val maskValue = -1e9
  /**
   * Create an bias tensor to be added to attention logits.
   * Returns tensor with shape (1, 1, length, length)
   * @param length
   * @tparam T
   * @return
   */
  def attentionBiasLowerTriangle[T: ClassTag](
     length: Int, output: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val arr = output.storage().array()
    for (i <- 0 to (length - 1)) {
      var j = length - 1
      while (j > i) {
        // reminder: here not 1
        arr(i * length + j) = ev.fromType(maskValue)
        j -= 1
      }
    }
    output.resize(Array(1, 1, length, length))
  }
}

sealed trait TransformerType

case object Translation extends TransformerType
case object LanguageModel extends TransformerType
