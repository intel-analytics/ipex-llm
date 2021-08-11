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

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
import com.intel.analytics.bigdl.utils.{Engine, Table}

import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * It is a table module which takes a table of Tensors as input and
 * outputs a Tensor by joining them together along the dimension `dimension`.
 *
 * The input to this layer is expected to be a tensor, or a batch of tensors;
 * when using mini-batch, a batch of sample tensors will be passed to the layer and
 * the user need to specify the number of dimensions of each sample tensor in the
 * batch using `nInputDims`.
 *
 * @param dimension to be join in this dimension
 * @param nInputDims specify the number of dimensions that this module will receive
 *                   If it is more than the dimension of input tensors, the first dimension
 *                   would be considered as batch size
 */

@SerialVersionUID(- 8435694717504118735L)
class JoinTable[T: ClassTag] (
  val dimension: Int,
  val nInputDims: Int
)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Tensor[_], T] {

  @transient
  private var results: Array[Future[Unit]] = null

  private def getPositiveDimension(input: Table): Int = {
    var nDim = this.dimension
    val firstInput: Tensor[_] = input(1)

    if (nDim < 0) {
      nDim = firstInput.dim() + nDim + 1
    } else if (nInputDims > 0 && firstInput.dim() == (nInputDims + 1)) {
      nDim += 1
    }
    require(firstInput.dim() >= dimension, "dimension exceeds input dimensions" +
      s" input dimension ${firstInput.dim()}, dimension ${dimension}")
    nDim
  }

  override def updateOutput(input: Table): Tensor[_] = {
    val dimension = getPositiveDimension(input)
    var size: Array[Int] = null

    var i = 1
    while (i <= input.length()) {
      val currentOutput: Tensor[_] = input(i)
      if (i == 1) {
        size = currentOutput.size()
      } else {
        size(dimension - 1) += currentOutput.size(dimension)
      }
      i += 1
    }
    val firstInput = input[Tensor[_]](1)
    if (output.getType() != firstInput.getType()) {
      output = firstInput.emptyInstance().resize(size)
    } else {
      output.resize(size)
    }

    if (results == null || results.length != input.length) {
      results = new Array[Future[Unit]](input.length)
    }
    var offset = 1
    i = 0
    while (i < input.length) {
      val currentOutput = input(i + 1).asInstanceOf[Tensor[NumericWildcard]]
      val _offset = offset
      results(i) = Engine.model.invoke( () => {
        val target = output.narrow(dimension, _offset, currentOutput.size(dimension))
          .asInstanceOf[Tensor[NumericWildcard]]
        if (target.isContiguous() || dimension > 2) {
          target.copy(currentOutput)
        } else {
          var f = 1
          while (f <= target.size(1)) {
            val curFrame = target.select(1, f)
            val outputFrame = currentOutput.select(1, f)
            require(curFrame.isContiguous())
            require(outputFrame.isContiguous())
            curFrame.copy(outputFrame)
            f += 1
          }
        }
      })
      i += 1
      offset += currentOutput.size(dimension)
    }
    Engine.model.sync(results)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[_]): Table = {
    val dimension = getPositiveDimension(input)

    var offset = 1
    var i = 0
    while (i < input.length) {
      val currentOutput = input(i + 1).asInstanceOf[Tensor[_]]
      val _offset = offset
      val _i = i
      results(i) = Engine.model.invoke( () => {
        val narrowedTensor = gradOutput.narrow(dimension, _offset, currentOutput.size(dimension))
          .asInstanceOf[Tensor[NumericWildcard]]
        val inputTensor = input[Tensor[_]](_i + 1)
        if (!gradInput.contains(_i + 1)) {
          gradInput(_i + 1) =
            inputTensor.emptyInstance().resizeAs(inputTensor)
        } else {
          gradInput[Tensor[T]](_i + 1).resizeAs(inputTensor)
        }
        if(narrowedTensor.isContiguous() || dimension > 2) {
          gradInput[Tensor[NumericWildcard]](_i + 1).copy(narrowedTensor)
        } else {
          var b = 1
          while(b <= narrowedTensor.size(1)) {
            val curFrame = gradInput[Tensor[_]](_i + 1).select(1, b)
              .asInstanceOf[Tensor[NumericWildcard]]
            val narrowFrame = narrowedTensor.select(1, b)
            require(curFrame.isContiguous())
            require(narrowFrame.isContiguous())
            curFrame.copy(narrowFrame)
            b += 1
          }
        }
      })
      i += 1
      offset += currentOutput.size(dimension)
    }
    Engine.model.sync(results)
    gradInput
  }

  override def toString: String = s"nn.JoinTable"


  override def canEqual(other: Any): Boolean = other.isInstanceOf[JoinTable[T]]

  override def equals(other: Any): Boolean = other match {
    case that: JoinTable[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        dimension == that.dimension &&
        nInputDims == that.nInputDims
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), dimension, nInputDims)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def clearState(): this.type = {
    super.clearState()
    gradInput.clear()
    this
  }
}

object JoinTable {
  def apply[@specialized(Float, Double) T: ClassTag](
      dimension: Int,
      nInputDims: Int)(implicit ev: TensorNumeric[T]) : JoinTable[T] = {
    new JoinTable[T](dimension, nInputDims)
  }
}
