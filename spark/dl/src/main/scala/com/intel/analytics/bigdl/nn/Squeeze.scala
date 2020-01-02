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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.utils.Shape
import scala.collection.mutable.ArrayBuffer

import scala.reflect.ClassTag

/**
 * Delete all singleton dimensions or a specific singleton dimension.
 *
 * @param dims Optional. If this dimension is singleton dimension, it will be deleted.
 *            The first index starts from 1. Default: delete all dimensions.
 * @param batchMode Optional. If the input is batch. Default is false.
 */

@SerialVersionUID(7998127436291978408L)
class Squeeze[T: ClassTag](
  val dims : Array[Int] = null, val batchMode: Boolean = false
  )(implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[_], Tensor[_], T]  {

  val dimensions = if (batchMode && dims != null) {
    val newDims = new Array[Int](dims.length)
    var i = 0
    while(i < newDims.length) {
      newDims(i) = dims(i) + 1
      i += 1
    }
    newDims
  } else {
    dims
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val _inputSize = inputShape.toSingle().toArray
    var resultSize = new ArrayBuffer[Int]()
    for (i <- 1 to _inputSize.length) {
      if (!dims.contains(i) || (dims.contains(i) && _inputSize(i - 1) != 1)) {
        resultSize.append(_inputSize(i - 1))
      }
    }
    Shape(resultSize.toArray)
  }

  override def updateOutput(input: Tensor[_]): Tensor[_] = {
    if (output.getType() != input.getType()) {
      output = input.emptyInstance()
    }
    output.asInstanceOf[Tensor[NumericWildcard]].set(input.asInstanceOf[Tensor[NumericWildcard]])
    if (dimensions != null) {
      var i = 0
      while(i < dimensions.length) {
        output.squeeze(dimensions(i))
        i += 1
      }
    } else {
      output.squeeze()
    }

    if (batchMode && dimensions == null && input.size(1) == 1) {
      output.addSingletonDimension()
    }
    output
  }

  override def updateGradInput(input: Tensor[_], gradOutput: Tensor[_]): Tensor[_] = {
    if (gradInput.getType() != gradOutput.getType()) {
      gradInput = gradOutput.emptyInstance()
    }
    require(input.nElement() == gradOutput.nElement(),
      "input and gradoutput shoule be of the same size" +
        s"input size ${input.nElement()} gradoutput size ${gradOutput.nElement()}")
    gradInput.asInstanceOf[Tensor[NumericWildcard]]
      .set(gradOutput.asInstanceOf[Tensor[NumericWildcard]].view(input.size()))
    gradInput
  }

  override def toString(): String = {
    s"${getPrintName}(${if (dimensions != null) dimensions.mkString(",") + ", " else ""}" +
      s"${if (batchMode) "batch" else ""})"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Squeeze[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Squeeze[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        (dims.zip(that.dimensions).map(a => a._1 == a._2).reduce(_ && _)) &&
        batchMode == that.batchMode
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), dimensions, batchMode)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object Squeeze {
  def apply[T: ClassTag](dim : Int = Int.MinValue,
    numInputDims: Int = Int.MinValue)(implicit ev: TensorNumeric[T])
  : Squeeze[T] = {
    new Squeeze[T](Array(dim), numInputDims != Int.MinValue)
  }

  def apply[T: ClassTag](
    dims : Array[Int], batchMode: Boolean)(implicit ev: TensorNumeric[T])
  : Squeeze[T] = {
    new Squeeze[T](if (dims != null) dims.sortWith(_>_) else null, batchMode)
  }
}
