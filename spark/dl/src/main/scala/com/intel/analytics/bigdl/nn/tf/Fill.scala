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

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.nn.Utils
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
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
    require(shapeTensor.nDimension() == 1, "shape tensor is not a vector")
    val shape = new Array[Int](shapeTensor.nElement())
    var i = 0
    while (i < shapeTensor.nElement()) {
      shape(i) = shapeTensor.valueAt(i + 1)
      i = i + 1
    }
    val value = input[Tensor[_]](2)
    require(value.isScalar, "value tensor is not a scalar")
    if (value.getType() != output.getType()) {
      output = Utils.allocate(shape, value.getType())
    } else {
      output.resize(shape)
    }

    fill(output, value)

    output
  }

  private def fill(output: Tensor[_], valueTensor: Tensor[_]): Unit = {
    require(output.getType() == valueTensor.getType(),
      "fill tensor and value tensor are not the same type")
    valueTensor.getType() match {
      case FloatType =>
        val value = valueTensor.asInstanceOf[Tensor[Float]].value()
        output.asInstanceOf[Tensor[Float]].fill(value)
      case DoubleType =>
        val value = valueTensor.asInstanceOf[Tensor[Double]].value()
        output.asInstanceOf[Tensor[Double]].fill(value)
      case IntType =>
        val value = valueTensor.asInstanceOf[Tensor[Int]].value()
        output.asInstanceOf[Tensor[Int]].fill(value)
      case LongType =>
        val value = valueTensor.asInstanceOf[Tensor[Long]].value()
        output.asInstanceOf[Tensor[Long]].fill(value)
      case ShortType =>
        val value = valueTensor.asInstanceOf[Tensor[Short]].value()
        output.asInstanceOf[Tensor[Short]].fill(value)
      case CharType =>
        val value = valueTensor.asInstanceOf[Tensor[Char]].value()
        output.asInstanceOf[Tensor[Char]].fill(value)
      case StringType =>
        val value = valueTensor.asInstanceOf[Tensor[ByteString]].value()
        output.asInstanceOf[Tensor[ByteString]].fill(value)
      case BooleanType =>
        val value = valueTensor.asInstanceOf[Tensor[Boolean]].value()
        output.asInstanceOf[Tensor[Boolean]].fill(value)
    }
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[_]): Table = {
    if (gradInput.contains(1)) {
      gradInput[Tensor[_]](1).resize(input[Tensor[_]](1).size()).zero()
    } else {
      gradInput(1) = Utils.allocate(input[Tensor[_]](1).size(), input[Tensor[_]](1).getType())
    }

    if (gradInput.contains(2)) {
      gradInput[Tensor[_]](2).resize(input[Tensor[_]](2).size()).zero()
    } else {
      gradInput(2) = Utils.allocate(input[Tensor[_]](2).size(), input[Tensor[_]](2).getType())
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
