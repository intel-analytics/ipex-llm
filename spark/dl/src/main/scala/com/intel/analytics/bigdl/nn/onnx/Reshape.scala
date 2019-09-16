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


package com.intel.analytics.bigdl.nn.onnx

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table


/**
 * Reshape the input tensor similar to numpy.reshape.
 * First input is the data tensor, second input is a shape tensor which specifies the output shape.
 * It outputs the reshaped tensor.
 * @param `classTag$T`
 * @param ev
 * @tparam T The numeric type in this module parameters.
 */
class Reshape[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Tensor[T], T] {

  override def updateOutput(input: Table): Tensor[T] = {
    require(input.length() == 2)
    val dataTensor: Tensor[T] = input.get[Tensor[T]](1).get
    val shape: Array[Int] = input.get[Tensor[T]](2).get.squeeze().toArray().map(ev.toType[Int])

    val innerReshaper = nn.Reshape(shape, batchMode = Option(false))

    output = innerReshaper.forward(dataTensor)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    gradInput = input
    gradInput
  }

}

object Reshape {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Reshape[T] = {
    new Reshape[T]()
  }
}
