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

  var concatPtr: Long = 0L
  var concat1Pass: Boolean = true

  var sumPtr: Long = 0L
  var sum1Pass: Boolean = true

  override def getClassPtr(): Long = concatPtr

  def getSize(): Array[Int] = {
    return size
  }

  override def reset(): Unit = {
    require(this.modules.length <= 4 && this.modules.length >= 1)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(this.modules.length <= 4 && this.modules.length >= 1)
    if (sum1Pass) {
      val nDimension = input.nDimension()
      val oneOutput: Array[Int] = new Array[Int](nDimension)

      for (j <- 0 until nDimension) {
        oneOutput(j) = input.size(nDimension - j)
      }

      ev.getType() match {
        case "Double" =>
          sumPtr = MKL.SumInitDouble(this.modules.length, nDimension, oneOutput)
        case "Float" =>
          sumPtr = MKL.SumInitFloat(this.modules.length, nDimension, oneOutput)
        case _ =>
          throw new UnsupportedOperationException(s"Only Float supported")
      }
      sum1Pass = false
    }

//    val sumOuts: Array[Tensor[T]] = new Array[Tensor[T]](this.modules.length)
//    val sumOutputs: Array[Array[T]] = new Array[Array[T]](this.modules.length)
//    val sumOutputsOffset: Array[Int] = new Array[Int](this.modules.length)
//    for (i <- 0 until this.modules.length) {
//      sumOuts(i) = Tensor[T]()
//      sumOuts(i).resizeAs(input)
//      sumOutputs(i) = sumOuts(i).storage().array()
//      sumOutputsOffset(i) = sumOuts(i).storageOffset() - 1
//    }
//
//    ev.getType() match {
//      case "Double" =>
//        MKL.SumForwardDouble(input.storage().array().asInstanceOf[Array[Double]],
//                             input.storageOffset() - 1,
//                             sumOutputs.asInstanceOf[Array[Array[Double]]],
//                             sumOutputsOffset,
//                             sumPtr)
//      case "Float" =>
//        MKL.SumForwardFloat(input.storage().array().asInstanceOf[Array[Float]],
//                            input.storageOffset() - 1,
//                            sumOutputs.asInstanceOf[Array[Array[Float]]],
//                            sumOutputsOffset,
//                            sumPtr)
//    }

    // TODO should check the size of every tensor. It must be same as the first tensor
    for (j <- 0 until this.modules.length) {
      if (initForward) {
        this.modules(j).setPrevPtr(this.getPrevPtr())
      }
    }
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
    if (concat1Pass) {
      // TODO we should not specify the dimension.
      val nDimension = outs(0).nDimension()
      val inputSize: Array[Int] = new Array[Int](this.modules.length * 4)

      // TODO should make it simple
      for (i <- 0 until this.modules.length) {
        for (j <- 0 until nDimension) {
          inputSize(i * 4 + 4 - nDimension + j) = outs(i).size(nDimension - j)
        }

        for (j <- 0 until (4 - nDimension)) {
          inputSize(i * 4 + j) = 1
        }
      }

      ev.getType() match {
        case "Double" =>
          concatPtr = MKL.ConcatInitDouble(this.modules.length, 4, inputSize)
        case "Float" =>
          concatPtr = MKL.ConcatInitFloat(this.modules.length, 4, inputSize)
        case _ =>
          throw new UnsupportedOperationException(s"Only Float supported")
      }
      concat1Pass = false
    }

    if (this.initForward) {
      this.updateMklOut()
      this.initForward = false
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
                                concatPtr)
      case "Float" =>
        MKL.ConcatForwardFloat(inputs.asInstanceOf[Array[Array[Float]]],
                               inputsOffset,
                               output.storage().array().asInstanceOf[Array[Float]],
                               output.storageOffset() - 1,
                               concatPtr)
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    this.output
  }

  // TODO should we implement this function, what's the difference from @backward
  // TODO this function must be implemented, and then the testcases in mkl should be changed,
  //      from backward -> updateGradInput.
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
    var totalSize: Long = 0L
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

    for (i <- 0 until this.modules.length) {
      this.modules(i).setNextPtr(this.modules(i).getOutputPtr())
    }

    val concatStart = System.nanoTime()
    ev.getType() match {
      case "Double" =>
        MKL.ConcatBackwardDouble(gradOutputs.asInstanceOf[Array[Array[Double]]],
                                 gradOutputsOffset,
                                 gradOutput.storage().array().asInstanceOf[Array[Double]],
                                 gradOutput.storageOffset() - 1,
                                 concatPtr)
      case "Float" =>
        MKL.ConcatBackwardFloat(gradOutputs.asInstanceOf[Array[Array[Float]]],
                                gradOutputsOffset,
                                gradOutput.storage().array().asInstanceOf[Array[Float]],
                                gradOutput.storageOffset() - 1,
                                concatPtr)
      case _ =>
        throw new UnsupportedOperationException(s"Only Float / Double is supported")
    }

    val concatEnd = System.nanoTime()

    val tmpGradInputs: Array[Tensor[T]] = new Array[Tensor[T]](this.modules.length)

    for (i <- 0 until this.modules.length) {
      val currentOutput = this.modules(i).output
      tmpGradInputs(i) = this.modules(i).backward(input, gradouts(i))
    }

    // It can't be converted to mkl dnn concat forward, becaus the size of all
    // gradient input is the same.
    // copy method here doesn't costs too much
    // TODO convert to eltwise
    //if (currentGradInput != null) {
    //  if (i == 0) {
    //    this.gradInput.copy(currentGradInput)
    //  } else {
    //    this.gradInput.add(currentGradInput)
    //  }
    //}

    val sumStart = System.nanoTime()
    val subGradInputs: Array[Array[T]] = new Array[Array[T]](this.modules.length)
    val subGradInputsOffset: Array[Int] = new Array[Int](this.modules.length)
    for (i <- 0 until this.modules.length) {
      subGradInputs(i) = tmpGradInputs(i).storage().array()
      subGradInputsOffset(i) = tmpGradInputs(i).storageOffset() - 1
    }

    ev.getType() match {
      case "Double" =>
        MKL.SumBackwardDouble(gradInput.storage().array().asInstanceOf[Array[Double]],
                             gradInput.storageOffset() - 1,
                             subGradInputs.asInstanceOf[Array[Array[Double]]],
                             subGradInputsOffset,
                             sumPtr)
      case "Float" =>
        MKL.SumBackwardFloat(gradInput.storage().array().asInstanceOf[Array[Float]],
                            gradInput.storageOffset() - 1,
                            subGradInputs.asInstanceOf[Array[Array[Float]]],
                            subGradInputsOffset,
                            sumPtr)
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    if (initBackward) {
      updateMklGradInput()
      initBackward = false
    }

    val sumEnd = System.nanoTime()
//    println("Concat costs " + (concatEnd - concatStart) / 1e6)
//    println("Sum costs " + (sumEnd - sumStart) / 1e6)

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

  // TODO we should use the next
  override def getInputPtr(): Long = sumPtr

  override def getOutputPtr(): Long = concatPtr

  override def updateMklOut(): Unit = {
    // If some layers are not mkl dnn version, we should set the previous layer
    // to convert the output based on layouts for scala.
    // Some notations:
    //
    // 1. Why it can work in the updateMklOut? Because the process of concat is
    //    that it will run submodules forward first, then do concat. And at the
    //    first time, the output of an layer will always be converted.
    val notInputAllMkl = this.modules.exists(_.getInputPtr() == 0)
    if (notInputAllMkl) {
      ev.getType() match {
        case "Double" => MKL.SetUseNextDouble(this.getPrevPtr(), 0)
        case "Float" => MKL.SetUseNextFloat(this.getPrevPtr(), 0)
      }
    }
    // Set the input of all concats.
    // println("CONCAT " + this.getName() + " " + this.concatPtr.toHexString)
    for (i <- 0 until this.modules.length) {
//      println("prev = " + this.modules(i).getOutputPtr().toHexString + " " + "CONCAT \tcurrent = " + this.concatPtr.toHexString)
      ev.getType() match {
        case "Double" =>
          MKL.SetConcatPrevDouble(this.modules(i).getOutputPtr(), i, this.concatPtr)
        case "Float" =>
          MKL.SetConcatPrevFloat(this.modules(i).getOutputPtr(), i, this.concatPtr)
        case _ =>
          throw new UnsupportedOperationException(s"Only support Float/Double")
      }
    }
  }

  override def updateMklGradInput(): Unit = {
    for (i <- 0 until this.modules.length) {
      ev.getType() match {
        case "Double" =>
          MKL.SetNextDouble(this.getNextPtr(), this.getOutputPtr())
        case "Float" =>
          MKL.SetNextFloat(this.getNextPtr(), this.getOutputPtr())
        case _ =>
          throw new UnsupportedOperationException(s"Only support Float/Double")
      }
    }

    // for concat
    for (i <- 0 until this.modules.length) {
      ev.getType() match {
        case "Double" =>
          MKL.SetConcatNextDouble(this.modules(i).getOutputPtr(), i, this.concatPtr)
        case "Float" =>
          MKL.SetConcatNextFloat(this.modules(i).getOutputPtr(), i, this.concatPtr)
        case _ =>
          throw new UnsupportedOperationException(s"Only support Float/Double")
      }
    }

    // for sum
    for (i <- 0 until this.modules.length) {
      ev.getType() match {
        case "Double" =>
          MKL.SetSumNextDouble(this.modules(i).getInputPtr(), i, this.sumPtr)
        case "Float" =>
          MKL.SetSumNextFloat(this.modules(i).getInputPtr(), i, this.sumPtr)
        case _ =>
          throw new UnsupportedOperationException(s"Only support Float/Double")
      }
    }
  }
}
