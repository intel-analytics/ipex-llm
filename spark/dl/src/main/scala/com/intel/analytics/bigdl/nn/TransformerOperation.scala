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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

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
    val res = getPadding[T](input)
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
    // todo: return shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    output.resizeAs(input).zero()
    val index = input.size(2)
    output.narrow(2, 2, index - 1).copy(input.narrow(2, 1, index - 1))
    output
  }
}
