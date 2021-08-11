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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.reflect.ClassTag

/**
 * Bottle allows varying dimensionality input to be forwarded through any module
 * that accepts input of nInputDim dimensions, and generates output of nOutputDim dimensions.
 *
 * @param module transform module
 * @param nInputDim nInputDim dimensions of module
 * @param nOutputDim1 output of nOutputDim dimensions
 */

@SerialVersionUID(8522437491532919144L)
class Bottle[T: ClassTag](
  val module: Module[T],
  val nInputDim: Int = 2,
  val nOutputDim1: Int = Int.MaxValue)
 (implicit ev: TensorNumeric[T]) extends DynamicContainer[Tensor[T], Tensor[T], T] {

  private val nOutputDim = if (nOutputDim1 == Int.MaxValue) nInputDim else nOutputDim1

  private val dimDelta = nInputDim - nOutputDim
  @transient
  private var inShape: Tensor[Double] = null
  @transient
  private var outShape: Tensor[Double] = null

  this.modules.insert(0, module)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    // first batchDims dimensions will be fused
    val batchDims = input.dim() - nInputDim + 1

    if (null == inShape) inShape = Tensor[Double](nInputDim)
    if (null == outShape) outShape = Tensor[Double](nOutputDim)

    if (batchDims > 1) {
      val inSize = Tensor[Double](Storage(input.size.map(_.toDouble)))

      val squeezeSize = inSize.storage().array().slice(0, batchDims - 1).product
      inShape.copy(inSize.narrow(1, batchDims, input.dim() - batchDims + 1))
      inShape.narrow(1, 1, 1).mul(squeezeSize)

      // Forward with the module's dimension
      val newInput = input.view(inShape.storage().array().map(_.toInt))
      val output1 = modules(0).forward(newInput).toTensor[T]
      require(output1.dim() == nOutputDim,
        s"Bottle: output dims on module should be $nOutputDim, but get ${output1.dim()}")

      outShape.copy(Tensor[Double](Storage(output1.size.map(_.toDouble))))

      if (math.abs(dimDelta) > 0) inSize.resize(inSize.size(1) - dimDelta)
      inSize.narrow(1, batchDims, inSize.size(1) - batchDims + 1).copy(outShape)
      inSize.narrow(1, batchDims, 1).div(squeezeSize)

      output.set(output1.view(inSize.storage().array().map(_.toInt)))
    } else {
      output.set(modules(0).forward(input).toTensor[T])
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (input.dim() > nInputDim) {
      val input_ = input.view(inShape.storage().array().map(_.toInt))
      val gradOutput_ = gradOutput.view(outShape.storage().array().map(_.toInt))
      modules(0).updateGradInput(input_, gradOutput_)
      val t2 = modules(0).gradInput.toTensor[T].resizeAs(input)
      gradInput.set(t2)
    } else {
      val t1 = modules(0).updateGradInput(input, gradOutput).toTensor[T]
      gradInput.set(t1)
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    if (input.dim() > nInputDim) {
      val input_ = input.view(inShape.storage().array().map(_.toInt))
      val gradOutput_ = gradOutput.view(outShape.storage().array().map(_.toInt))
      modules(0).accGradParameters(input_, gradOutput_)
    } else {
      modules(0).accGradParameters(input, gradOutput)
    }
  }

  override def toString(): String = {
    s"${getPrintName}($module, $nInputDim, $nOutputDim1)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Bottle[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Bottle[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        module == that.module &&
        nInputDim == that.nInputDim &&
        nOutputDim1 == that.nOutputDim1
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), module, nInputDim, nOutputDim1)
    state.map(getHashCode).foldLeft(0)((a, b) => 37 * a + b)
  }
}

object Bottle {
  def apply[@specialized(Float, Double) T: ClassTag](
    module: Module[T],
    nInputDim: Int = 2,
    nOutputDim1: Int = Int.MaxValue)(implicit ev: TensorNumeric[T]) : Bottle[T] = {
    new Bottle[T](module, nInputDim, nOutputDim1)
  }
}
