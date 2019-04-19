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
import com.intel.analytics.bigdl.nn.{Sequential, _}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

private[rnn] object TransformerOperation {
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
}
