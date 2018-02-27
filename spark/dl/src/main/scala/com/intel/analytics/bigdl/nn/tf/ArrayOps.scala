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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.ops.{ModuleToOperation, Operation}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildCard, TensorNumeric}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Some operation may not have input
 */
private[bigdl] trait WithoutInput

private[bigdl] class Const[T: ClassTag, B: ClassTag](val value: Tensor[B])
  (implicit ev: TensorNumeric[T])
  extends Operation[Activity, Tensor[B], T] with WithoutInput {

  override def updateOutput(input: Activity): Tensor[B] = {
    output = value
    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[B]),
      Array[TensorNumeric[_]](ev))
  }
}

private[bigdl] object Const {
  def apply[T: ClassTag, B: ClassTag](value: Tensor[B])
    (implicit ev: TensorNumeric[T]): Const[T, B] = {
    new Const[T, B](value)
  }
}

/**
 * This operation computes the inverse of an index permutation. It takes a 1-D integer tensor x,
 * which represents the indices of a zero-based array, and swaps each value with its index position.
 * In other words, for an output tensor y and an input tensor x, this operation computes the
 * following:
 *     y[x[i]] = i for i in [0, 1, ..., len(x) - 1]
 * The values must include 0. There can be no duplicate values or negative values.
 *
 * @tparam T Parameter numeric type. Only support float/double now
 */
private[bigdl] class InvertPermutation[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Operation[Tensor[Int], Tensor[Int], T] {

  output = Tensor[Int]()

  override def updateOutput(input: Tensor[Int]): Tensor[Int] = {
    require(input.dim() == 1, "InvertPermutation only accept 1D tensor as input")
    output.resizeAs(input)
    var i = 0
    while(i < input.size(1)) {
      output.setValue(input.valueAt(i + 1) + 1, i)
      i += 1
    }

    output
  }
}

/**
 * Calculate the positions of the input shapes in a concatenation operation. It takes several
 * tensors as input. The first tensor must be a scalar, which indicts on which dimension do the
 * concatenation. The offset of the dimension is start from zero.
 *
 * The left tensors must be 1D, as they represent the shapes of tensors. And they must be same
 * except the concat dimension.
 *
 * Her's an example, say, we want to concatenate 3 tensors in the 2nd dimension, the input shape
 * tensors should be
 *   [2, 2, 5, 7]
 *   [2, 3, 5, 7]
 *   [2, 4, 5, 7]
 *
 * The output should be
 *   [0, 0, 0, 0]
 *   [0, 2, 0, 0]
 *   [0, 5, 0, 0]
 * @tparam T Parameter numeric type. Only support float/double now
 */
private[bigdl] class ConcatOffset[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Operation[Table, Table, T] {

  output = T()

  override def updateOutput(input: Table): Table = {
    val concatDim = input[Tensor[Int]](1)
    require(concatDim.isScalar, "ConcatOffset: concat dim must be a scalar")
    val cdim = concatDim.value()
    val n = input.length() - 1
    var i = 1
    var offset = 0
    while(i <= n) {
      val shape = input[Tensor[Int]](i + 1)
      require(shape.nDimension() == 1, "ConcatOffset: shape must be 1D tensor")
      if (!output.contains(i)) {
        output(i) = Tensor[Int]()
      }
      val outputOffset = output[Tensor[Int]](i)
      outputOffset.resizeAs(shape).zero()
      outputOffset.setValue(cdim + 1, offset)
      val dimSize = shape.valueAt(cdim + 1)
      offset += dimSize
      i += 1
    }

    output
  }
}

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

/**
 * Given shapes of two tensors, computes the reduction indices for the
 * gradient computation.
 *
 * @tparam T Numeric type. Only support float/double now
 */
private[bigdl] class BroadcastGradientArgs[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Operation[Table, Table, T] {

  override def updateOutput(input: Table): Table = {
    val input1 = input[Tensor[Int]](1)
    val input2 = input[Tensor[Int]](2)

    val output1 = Tensor[Int]()
    val output2 = Tensor[Int]()

    output.insert(output1).insert(output2)

    // Reverse the shape of x and y for convenience.
    // After the reverse, 0-th is the inner-most dimension.
    val rx =
    if (input1.storage() == null) Array[Int]().toBuffer
    else input1.storage().array().reverse.toBuffer
    val ry =
      if (input2.storage() == null) Array[Int]().toBuffer
      else input2.storage().array().reverse.toBuffer

    if (rx.length < ry.length) {
      while (rx.length < ry.length) {
        rx.append(1)
      }
    } else {
      while (rx.length > ry.length) {
        ry.append(1)
      }
    }

    val xReducedIndexBuffer = new ArrayBuffer[Int]()
    val yReducedIndexBuffer = new ArrayBuffer[Int]()

    val n = rx.length

    var i = 0
    while (i < n) {
      val xi = rx(i)
      val yi = ry(i)

      if (xi == yi) {
        if (xi == 1) {
          xReducedIndexBuffer.append(n - 1 - i)
          yReducedIndexBuffer.append(n - 1 - i)
        }
      } else if (xi == 1) {
        xReducedIndexBuffer.append(n - 1 - i)
      } else if (yi == 1) {
        yReducedIndexBuffer.append(n - 1 - i)
      } else {
        return output
      }
      i += 1
    }

    if (xReducedIndexBuffer.isEmpty) {
      input(1) = Tensor[Int]()
    } else {
      output1.resize(Array(xReducedIndexBuffer.length))
        .set(Tensor[Int](Storage(xReducedIndexBuffer.reverse.toArray)))
    }

    if (yReducedIndexBuffer.isEmpty) {
      input(2) = Tensor[Int]()
    } else {
      output2.resize(Array(yReducedIndexBuffer.length))
        .set(Tensor[Int](Storage(yReducedIndexBuffer.reverse.toArray)))
    }

    output
  }
}

private[bigdl] object BroadcastGradientArgs {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](new BroadcastGradientArgs())
}
