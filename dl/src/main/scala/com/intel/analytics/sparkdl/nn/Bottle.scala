/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import com.intel.analytics.sparkdl.utils.Activities

import scala.reflect.ClassTag

class Bottle[A <: Activities : ClassTag, B <: Activities : ClassTag, T: ClassTag]
(module: Module[A, B, T], nInputDim: Int = 2, nOutputDim: Int)
(implicit ev: TensorNumeric[T]) extends Container[A, B, T] {

  // nOutputDim = if (nOutputDim == null) nInputDim else nInputDim
  val dimDelta = nInputDim - nOutputDim

  // Used to reshape the gradients
  val inShape = Tensor[Double](nInputDim)
  val outShape = Tensor[Double](nOutputDim)

  // add module to modules
  // this.modules(0) = module.asInstanceOf[Module[Activities, Activities, T]]

  def arrayIntToDouble(data : Array[Int]): Array[Double] = {
    var i = 0
    val res = new Array[Double](data.length)
    while (i < data.length) {
      res(i) = data(i).toDouble
      i += 1
    }
    res
  }

  def arrayDoubleToInt(data : Array[Double]): Array[Int] = {
    var i = 0
    val res = new Array[Int](data.length)
    while (i < data.length) {
      res(i) = data(i).toInt
      i += 1
    }
    res
  }

  override def updateOutput(input: A): B = {
    // first batchDims dimensions will be fused
    this.modules(0) = module.asInstanceOf[Module[Activities, Activities, T]]
    val res = input.toTensor[T]()
    val batchDims = res.dim() - nInputDim + 1
    if (batchDims > 1) {
      val inSize = Tensor[Double](Storage(arrayIntToDouble(res.size)))

      val squeezeSize = inSize.storage().array().slice(0, batchDims - 1).product
      inShape.copy(inSize.narrow(1, batchDims, res.dim() - batchDims + 1))
      inShape.narrow(1, 1, 1).mul(squeezeSize)

      // Forward with the module's dimension
      val newInput = res.view(arrayDoubleToInt(inShape.storage().array()))
      val output1 = modules(0).updateOutput(newInput).toTensor[T]()
      require(output1.dim() == nOutputDim, "Wrong number of output dims on module")

      outShape.copy(Tensor[Double](Storage(arrayIntToDouble(output1.size))))

      if (math.abs(dimDelta) > 0) inSize.resize(inSize.size(1) - dimDelta)
      println(batchDims)
      println(inSize.size(1) - batchDims)
      inSize.narrow(1, batchDims, inSize.size(1) - batchDims + 1).copy(outShape)
      inSize.narrow(1, batchDims, 1).div(squeezeSize)

      output.asInstanceOf[Tensor[T]].
        set(output1.view(arrayDoubleToInt(inSize.storage().array())))
    } else {
      output.asInstanceOf[Tensor[T]].
        set(modules(0).updateOutput(input.toTensor[T]()).toTensor[T]())
    }
    output
  }

  override def updateGradInput(input: A, gradOutput: B): A = {
    if (input.toTensor().dim() > nInputDim) {
      val input_ = input.toTensor().view(arrayDoubleToInt(inShape.storage().array()))
      val gradOutput_ = gradOutput.toTensor().view(arrayDoubleToInt(outShape.storage().array()))
      modules(0).updateGradInput(input_, gradOutput_)
      val t2 = modules(0).gradInput.toTensor[T]().resizeAs(input.toTensor())
      gradInput.asInstanceOf[Activities].toTensor[T]().set(t2)
    } else {
      val t1 = modules(0).updateGradInput(input.asInstanceOf[Activities],
        gradOutput.asInstanceOf[Activities]).toTensor[T]()
      gradInput.asInstanceOf[Activities].toTensor[T]().set(t1)
    }
    gradInput
  }

  override def accGradParameters(input: A, gradOutput: B, scale: Double): Unit = {
    if (input.toTensor().dim() > nInputDim) {
      val input_ = input.toTensor().view(arrayDoubleToInt(inShape.storage().array()))
      val gradOutput_ = gradOutput.toTensor().view(arrayDoubleToInt(outShape.storage().array()))
      modules(0).accGradParameters(input_, gradOutput_, scale)
    } else {
      modules(0).accGradParameters(input, gradOutput, scale)
    }
  }

  override def toString(): String = {
    s"nn.Bottle"
  }
}
