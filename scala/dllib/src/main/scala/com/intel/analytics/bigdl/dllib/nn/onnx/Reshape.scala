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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
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
class Reshape[T: ClassTag](var shape: Array[Int] = null)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Activity, Tensor[T], T] {

  override def updateOutput(input: Activity): Tensor[T] = {
    var dataTensor: Tensor[T] = null

    if (input.isTable) {
      val inputTable = input.toTable
      require(inputTable.length() == 2)
      dataTensor = inputTable.get[Tensor[T]](1).get
      shape = inputTable.get[Tensor[T]](2).get.squeeze().toArray().map(ev.toType[Int])
    } else if (input.isTensor) {
      dataTensor = input.toTensor[T]
    } else {
      throw new IllegalArgumentException()
    }
    require(shape != null, "shape should not be null")
    val innerReshaper = nn.Reshape(shape, batchMode = Option(false))
    output = innerReshaper.forward(dataTensor)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Tensor[T]): Activity = {
    val inputTensor = if (input.isTable) {
      input.toTable.get[Tensor[T]](1).get
    } else if (input.isTensor) {
      input.toTensor[T]
    } else {
      throw new IllegalArgumentException()
    }
    gradInput = inputTensor.zero()
    gradInput
  }

}


object Reshape {
  def apply[T: ClassTag](shape: Array[Int] = null)
    (implicit ev: TensorNumeric[T]): Reshape[T] = {
    new Reshape[T](shape)
  }
}
