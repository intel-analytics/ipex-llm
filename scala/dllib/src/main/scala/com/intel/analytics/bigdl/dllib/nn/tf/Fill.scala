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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildCard, TensorNumeric}
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Creates a tensor filled with a scalar value. Input should be a 1-D tensor defining
 * the shape of the output tensor.
 */
@SerialVersionUID(-471757174144422555L)
private[bigdl] class Fill[T: ClassTag]() (implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Tensor[_], T] {

  override def updateOutput(input: Table): Tensor[_] = {
    val shapeTensor = input[Tensor[Int]](1)
    val value = input[Tensor[_]](2)
    if (shapeTensor.isEmpty) {
      if (value.getType() != output.getType()) {
        output = value.emptyInstance()
      }
      output.resizeAs(value).asInstanceOf[Tensor[NumericWildCard]]
        .copy(value.asInstanceOf[Tensor[NumericWildCard]])
    } else {
      require(shapeTensor.nDimension() == 1, "shape tensor is not a vector")
      val shape = new Array[Int](shapeTensor.nElement())
      var i = 0
      while (i < shapeTensor.nElement()) {
        shape(i) = shapeTensor.valueAt(i + 1)
        i = i + 1
      }
      require(value.isScalar, "value tensor is not a scalar")
      if (value.getType() != output.getType()) {
        output = value.emptyInstance().resize(shape)
      } else {
        output.resize(shape)
      }

      output.forceFill(value.value())
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[_]): Table = {
    if (gradInput.contains(1)) {
      gradInput[Tensor[_]](1).resize(input[Tensor[_]](1).size()).zero()
    } else {
      val inputTensor = input[Tensor[_]](1)
      gradInput(1) = inputTensor.emptyInstance().resize(inputTensor.size())
    }

    if (gradInput.contains(2)) {
      gradInput[Tensor[_]](2).resize(input[Tensor[_]](2).size()).zero()
    } else {
      val inputTensor = input[Tensor[_]](2)
      gradInput(2) = inputTensor.emptyInstance().resize(inputTensor.size())
    }
    gradInput
  }

}

private[bigdl] object Fill {
  def apply[T: ClassTag]()
       (implicit ev: TensorNumeric[T]) : Fill[T] = {
    new Fill[T]()
  }
}
