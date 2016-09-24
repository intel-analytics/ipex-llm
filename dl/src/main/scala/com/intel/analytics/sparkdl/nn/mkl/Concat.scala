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

/*
 * ATTENTION: MKL version. The start and end layer must be MKL version too.
 *            Currently, it supports BatchNormalization, Linear, LRN, Pooling(Avg, Max),
 *            ReLU and SpatialConvolution.
 */

package com.intel.analytics.sparkdl.nn.mkl

import com.intel.analytics.sparkdl.nn.{Container, Module}
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.mkl.MKL

import scala.reflect.ClassTag

class Concat[T: ClassTag](val dimension: Int)(implicit ev: TensorNumeric[T]) extends Container[T] {

  private var size: Array[Int] = null
  private var gradouts: Array[Tensor[T]] = null
  private var gradOutputs: Array[Array[T]] = Array[Array[T]]()

  var classPtr : Long = 0L
  var firstPass: Boolean = true

  override def getClassPtr(): Long = classPtr

  def getSize(): Array[Int] = {
    return size
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    // TODO should check the size of every tensor. It must be same as the first tensor
    val outs = new Array[Tensor[T]](this.modules.length)
    var i = 0
    while (i < this.modules.length) {
      val currentOutput = this.modules(i).updateOutput(input)
      outs(i) = currentOutput
      if (i == 0) {
        this.size = currentOutput.size()
      } else {
        this.size(this.dimension - 1) += currentOutput.size(this.dimension)
      }
      i += 1
    }

    this.output.resize(this.size)
    // TODO call mkl native code to update output
    // TODO dimension here is different with "dimension" in MKL 2017
    // TODO check all dimensions of input tensors are same
    if (firstPass) {
      val nDimension = outs(0).nDimension()
      val inputSize: Array[Int] = new Array[Int](this.modules.length * nDimension)

      for (i <- 0 until this.modules.length) {
        for (j <- 0 until nDimension) {
          inputSize(i * nDimension + j) = outs(i).size(nDimension - j)
        }
      }

      ev.getType() match {
        case "Double" =>
          classPtr = MKL.ConcatInitDouble(this.modules.length, nDimension, inputSize)
        case "Float" =>
          classPtr = MKL.ConcatInitFloat(this.modules.length, nDimension, inputSize)
        case _ =>
          throw new UnsupportedOperationException(s"Only Float supported")
      }
      firstPass = false
    }

    // get all of the tensors in outs to float/double array
    val inputs: Array[Array[T]] = new Array[Array[T]](this.modules.length)
    val inputsOffset: Array[Int] = new Array[Int](this.modules.length)
    for (i <- 0 until this.modules.length) {
      inputs(i) = outs(i).storage().array()
      inputsOffset(i) = outs(i).storageOffset() - 1
    }


    ev.getType() match {
      case "Double" =>
        MKL.ConcatForwardDouble(inputs.asInstanceOf[Array[Array[Double]]],
                                inputsOffset,
                                output.storage().array().asInstanceOf[Array[Double]],
                                output.storageOffset() - 1,
                                classPtr)
      case "Float" =>
        MKL.ConcatForwardFloat(inputs.asInstanceOf[Array[Array[Float]]],
                               inputsOffset,
                               output.storage().array().asInstanceOf[Array[Float]],
                               output.storageOffset() - 1,
                               classPtr)
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    this.output
  }

  // TODO should we implement this function, what's the difference from @backward
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
//    this.gradInput.resizeAs(input)
//
//    var offset = 1
//    var i = 0
//    while (i < this.modules.length) {
//      val currentOutput = this.modules(i).output
//      val currentGradInput = this.modules(i).updateGradInput(input,
//        gradOutput.narrow(dimension, offset, currentOutput.size(dimension)))
//
//      if (currentGradInput != null) {
//        if (i == 0) {
//          this.gradInput.copy(currentGradInput)
//        } else {
//          this.gradInput.add(currentGradInput)
//        }
//      }
//      i += 1
//      offset += currentOutput.size(dimension)
//    }

    this.gradInput
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    // TODO call mkl native code to update gradient input
    var totalSize : Long = 0L
    this.gradInput.resizeAs(input)
    if (gradouts == null || gradouts.length != this.modules.length) {
      gradouts = new Array[Tensor[T]](this.modules.length)
    }
    val gradOutputs: Array[Array[T]] = new Array[Array[T]](this.modules.length)
    val gradOutputsOffset: Array[Int] = new Array[Int](this.modules.length)
    for (i <- 0 until this.modules.length) {
      if (gradouts(i) == null) gradouts(i) = Tensor()
      gradouts(i).resizeAs(this.modules(i).output)
      gradOutputs(i) = gradouts(i).storage().array()
      gradOutputsOffset(i) = gradouts(i).storageOffset() - 1
    }

    ev.getType() match {
      case "Double" =>
        MKL.ConcatBackwardDouble(gradOutputs.asInstanceOf[Array[Array[Double]]],
                                 gradOutputsOffset,
                                 gradOutput.storage().array().asInstanceOf[Array[Double]],
                                 gradOutput.storageOffset() - 1,
                                 classPtr)
      case "Float" =>
        MKL.ConcatBackwardFloat(gradOutputs.asInstanceOf[Array[Array[Float]]],
                                gradOutputsOffset,
                                gradOutput.storage().array().asInstanceOf[Array[Float]],
                                gradOutput.storageOffset() - 1,
                                classPtr)
      case _ =>
        throw new UnsupportedOperationException(s"Only Float / Double is supported")
    }

    for (i <- 0 until this.modules.length) {
      val currentOutput = this.modules(i).output
      val currentGradInput = this.modules(i).backward(input, gradouts(i))

      // It can't be converted to mkl dnn concat forward, becaus the size of all
      // gradient input is the same.
      // copy method here doesn't costs too much
      // TODO convert to eltwise
      if (currentGradInput != null) {
        if (i == 0) {
          this.gradInput.copy(currentGradInput)
        } else {
          this.gradInput.add(currentGradInput)
        }
      }
    }

    this.gradInput
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[Concat[T]]) {
      return false
    }
    val other = obj.asInstanceOf[Concat[T]]
    if (this.eq(other)) {
      return true
    }
    if (dimension != other.dimension) {
      return false
    }

    if (this.modules.length != other.modules.length) {
      return false
    }

    val moduleLength = modules.length
    var i = 0
    while (i < moduleLength) {
      if (modules(i) != other.modules(i)) {
        return false
      }
      i += 1
    }

    true
  }
  override def hashCode(): Int = {

    val seed = 37
    var hash = super.hashCode()
    var i = 0
    val moduleLength = modules.length
    while (i < moduleLength) {
      hash = hash * seed + modules(i).hashCode()
      i += 1
    }

    hash
  }

  override def toString(): String = {
    val tab = "  "
    val next = "  |`-> "
    val last = "   ... -> "
    val ext = "  |    "
    val extlast = "       "
    s"mkl.Concat {$line${tab}input$line${modules.zipWithIndex.map {
      case (model: Module[T], index: Int) =>
        s"$tab$next(${index + 1}): ${if (index == modules.length - 1) {
          model.setLine(line + tab + extlast)
        } else {
          model.setLine(line + tab + ext)
        }}"
    }.mkString(line)}$line$tab${last}output$line$tab}"
  }
}
