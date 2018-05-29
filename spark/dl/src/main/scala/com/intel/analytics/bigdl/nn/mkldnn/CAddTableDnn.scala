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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl.{MKL, Memory}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.{MklDnnTensor, MklDnnType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect._

/**
 * Merge the input tensors in the input table by element wise adding them together. The input table
 * is actually an array of tensor with same size.
 * @param inplace reuse the input memory
 * @param ev numeric operator
 * @tparam T Numeric type. Only support float/double now
 */
@SerialVersionUID(7959261460060075605L)
class CAddTableDnn[T: ClassTag](val inplace: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  // no related with format
  override def updateOutput(input: Table): Tensor[T] = {
    val s1 = System.nanoTime()
    if (inplace) {
      output = input[Tensor[T]](1)
    } else if (output.getTensorType != MklDnnType) {
      output = MklDnnTensor[T](input[Tensor[T]](1).size())
      Memory.Zero(output.asInstanceOf[MklDnnTensor[T]].ptr, output.nElement(), 4)
    }

    // default: all tensor in inputTable should have same format
    val format : Int = input[Tensor[T]](1).getFormat()
    var i = if (inplace) { 2 } else { 1 }
    while (i <= input.length()) {
      val curTensor = input[Tensor[T]](i)
      require(curTensor.getFormat() == format, "all tensor in inputTable should have same format")
       // todo: need mkl dnn tensor add
      output.add(curTensor)
      i += 1
    }
    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      DnnTools.debugFwInfo(this.getName(), end1, -2, format)
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]) : Table = {
    val s1 = System.nanoTime()
    var i = 1
    var sum = ev.zero
    var calculateSum = false

    require(inplace == true, s"just supoort inplace=true ${this.getName()}")
    while (i <= input.length()) {
      if (i > gradInput.length) gradInput.insert(i, Tensor[T]().resizeAs(input(1)))
      if (inplace) {
        require(input[Tensor[T]](1).isSameSizeAs(gradOutput), "cannot use inplace for broadcast")
        gradInput(i) = gradOutput
      }
      i += 1
    }
    i = input.length + 1
    while (i <= gradInput.length) {
      gradInput.remove(i)
    }
    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      DnnTools.debugBwInfo(this.getName(), end1, gradOutput.getFormat(), -2)
    }
    gradInput
  }

  override def clearState(): this.type = {
    if (!inplace) {
      super.clearState()
    }
    this
  }
}


object CAddTableDnn {
  def apply[@specialized(Float, Double) T: ClassTag](
      inplace: Boolean = false)(implicit ev: TensorNumeric[T]) : CAddTableDnn[T] = {
    new CAddTableDnn[T](inplace)
  }
}


