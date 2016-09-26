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

import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag
import com.intel.analytics.sparkdl.utils.Engine

class Concat[@specialized(Float, Double) T: ClassTag] (val dimension : Int)(implicit ev: TensorNumeric[T]) extends Container[T]{
  private var size : Array[Int] = null
  private var results : Array[Future[_]] = null
  private var gradouts : Array[Tensor[T]] = null

  def getSize(): Array[Int] ={
    return size
  }

  def concatCopy[@specialized(Float, Double) T](tensor1 : Tensor[T], tensor2 : Tensor[T]) : Boolean = {
    require(tensor1.nElement() == tensor2.nElement(), "inconsistent tensor size")

    if (tensor1.nDimension == 0)
      return false

    val tensor1Stride = getStride(tensor1)
    val tensor2Stride = getStride(tensor2)

    if (tensor1Stride != 1 || tensor2Stride != 1) return false

    val (largestDim1, largestSize1) = getLargestContiguousSize(tensor1)
    val counter1 = getCounter(largestDim1)
    val (largestDim2, largestSize2) = getLargestContiguousSize(tensor2)
    val counter2 = getCounter(largestDim2)


    val tensor1Data = tensor1.storage().asInstanceOf[Storage[T]].array()
    var tensor1Offset = tensor1.storageOffset() - 1
    val tensor2Data = tensor2.storage().asInstanceOf[Storage[T]].array()
    var tensor2Offset = tensor2.storageOffset() - 1

    var adjacent = false
    if (tensor1.nDimension == 1 && tensor2.nDimension == 1 && tensor1.stride(1) == 1 && tensor2.stride(1) == 1) {
      adjacent = true
    }
    if (tensor1.nDimension == 2 && tensor2.nDimension == 2) {
      if (tensor1.stride(2) == 1 && tensor2.stride(2) == 1 && tensor1.stride(1) == tensor1.size(2) && tensor2.stride(1) == tensor2.size(2)) {
        adjacent = true
      }

      if (tensor1.stride(1) == 1 && tensor2.stride(1) == 1 && tensor1.stride(2) == tensor1.size(1) && tensor2.stride(2) == tensor2.size(1)) {
        adjacent = true
      }
    }
    if (adjacent) {
      System.arraycopy(tensor1Data, tensor1Offset, tensor2Data, tensor2Offset, tensor1.nElement())
      return true
    }

    /*
    if (tensor1Stride != 1 || tensor2Stride != 1) {
      println("tessor1Stride = " + tensor1Stride)
      println("tessor2Stride = " + tensor2Stride)
    }

    if (largestDim1 != 0 || largestDim2 != 0) {
      println("largestDim1 = " + largestDim1)
      println("largestSize1 = " + largestSize1)
      println("largestDim2 = " + largestDim2)
      println("largestSize2 = " + largestSize2)
    }
    */

    /*
    if (tensor1Stride == 1 && tensor2Stride == 1) {
      var hasFinished = false
      val copyNum = if (largestSize1 < largestSize2) largestSize1 else largestSize2
      System.arraycopy(tensor1Data, tensor1Offset, tensor2Data, tensor2Offset, copyNum)
    }
    */

    var hasFinished = false
    var i1 = 0
    var i2 = 0
    while (!hasFinished) {
      val start = System.nanoTime()
      val copyNum = if (largestSize1 < largestSize2) largestSize1 else largestSize2
      System.arraycopy(tensor1Data, tensor1Offset, tensor2Data, tensor2Offset, copyNum)
      i1 += copyNum
      i2 += copyNum
      // println("[" + Thread.currentThread().getName() + "]" + " concat-copy array " + (System.nanoTime() - start) / 1e6)

      if (i1 == largestSize1) {
        val r = updateCounter(tensor1, counter1, tensor1Offset, largestDim1)
        hasFinished = r._1
        tensor1Offset = r._2
        i1 = 0
      }

      if (i2 == largestSize2) {
        val r = updateCounter(tensor2, counter2, tensor2Offset, largestDim2)
        hasFinished = r._1
        tensor2Offset = r._2
        i2 = 0
      }
    }

    return true
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val outs = new Array[Tensor[T]](this.modules.length)
    var i = 0
    while(i < this.modules.length) {
      val currentOutput = this.modules(i).updateOutput(input)
      outs(i) = currentOutput
      if(i == 0) {
        this.size = currentOutput.size()
      } else {
        this.size(this.dimension - 1) += currentOutput.size(this.dimension)
      }
      i += 1
    }

    this.output.resize(this.size)
    if(results == null || results.length != this.modules.length) {
      results = new Array[Future[_]](this.modules.length)
    }

    val start = System.nanoTime()
    var offset = 1
    i = 0
    var copyTime = 0L
    var selectTime = 0L
    var startc = 0L
    while(i < this.modules.length) {
      val currentOutput = outs(i)
      val _offset = offset
      results(i) = Future {
        val start = System.nanoTime()
        val target = this.output.narrow(this.dimension, _offset, currentOutput.size(this.dimension))//.copy(currentOutput)
        // println("[SCALA] [" + Thread.currentThread().getName() + "]" + " concat-narrow after module forward costs " + (System.nanoTime() - start) / 1e6)
        var f = 1
        while(f <= target.size(1)) {
          startc = System.nanoTime()
          val curFrame = target.select(1, f)
          val outputFrame = currentOutput.select(1, f)
          selectTime += System.nanoTime() - startc
          require(curFrame.isContiguous())
          require(outputFrame.isContiguous())
          startc = System.nanoTime()
          if (!concatCopy(curFrame, outputFrame)) {
            println("STRIDE NOT EQUAL 1")
            curFrame.copy(outputFrame)
          }
          copyTime += (System.nanoTime() - startc)
          f += 1
        }
      }(Engine.getInstance())
      i += 1
      offset += currentOutput.size(this.dimension)
    }

    i = 0
    while(i < results.length) {
      Await.result(results(i), Duration.Inf)
      i += 1
    }
    // println("[SCALA] [" + Thread.currentThread().getName + "]" + " concat-loop-copy after module forward costs " + copyTime / 1e6)
    // println("[SCALA] [" + Thread.currentThread().getName + "]" + " concat-loop-select after module forward costs " + selectTime / 1e6)

    val end = System.nanoTime()
    //println("[SCALA] [" + Thread.currentThread().getName + "]" + " concat after module forward costs " + (end-start)/1e6)

    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    this.gradInput.resizeAs(input)

    var offset = 1
    var i = 0
    while(i < this.modules.length) {
      val currentOutput = this.modules(i).output
      val currentGradInput = this.modules(i).updateGradInput(input,
        gradOutput.narrow(dimension, offset, currentOutput.size(dimension)))

      if(currentGradInput != null) {
        if(i == 0) {
          if (!concatCopy(this.gradInput, currentGradInput)) {
            println("STRIDE NOT EQUAL 1")
            this.gradInput.copy(currentGradInput)
          }
        } else {
          this.gradInput.add(currentGradInput)
        }
      }
      i += 1
      offset += currentOutput.size(dimension)
    }

    this.gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double = 1.0): Unit = {
    var offset = 1
    var i = 0
    while(i < this.modules.length) {
      val currentOutput = this.modules(i).output
      this.modules(i).accGradParameters(
        input,
        gradOutput.narrow(dimension, offset, currentOutput.size(dimension)),
        scale)

      i += 1
      offset += currentOutput.size(dimension)
    }
  }

  def concatContiguous(tensor1: Tensor[T], tensor2: Tensor[T]) : Boolean = {
    if (!tensor2.isContiguous()) {
      tensor1.resizeAs(tensor2)
      return concatCopy(tensor1, tensor2)
    } else {
      return false
    }
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val start = System.nanoTime()
    val before = System.nanoTime()
    this.gradInput.resizeAs(input)
    var offset = 1
    if(gradouts == null || gradouts.length != this.modules.length) {
      gradouts = new Array[Tensor[T]](this.modules.length)
    }
    var i = 0
    while(i < this.modules.length) {
      val currentOutput = this.modules(i).output
      val _offset = offset
      val _i = i
      results(i) = Future {
        // gradouts(_i) = gradOutput.narrow(dimension, _offset, currentOutput.size(dimension)).contiguous()
        val tmpTensor =gradOutput.narrow(dimension, _offset, currentOutput.size(dimension))
        if (!tmpTensor.isContiguous()) {
          gradouts(_i) = Tensor[T]()
          gradouts(_i).resizeAs(tmpTensor)
          val ret = concatCopy(gradouts(_i), tmpTensor)
        } else {
          gradouts(_i) = tmpTensor
        }
      }(Engine.getInstance())
      i += 1
      offset += currentOutput.size(dimension)
    }
    i = 0
    while(i < this.modules.length) {
      Await.result(results(i), Duration.Inf)
      i += 1
    }
    val end = System.nanoTime()
    //println("[SCALA] [" + Thread.currentThread().getName() + "]" + "  concat before module backward costs " + (end - start) / 1e6)

    i = 0
    offset = 1
    while(i < this.modules.length) {
      val currentOutput = this.modules(i).output
      val currentGradInput = this.modules(i).backward(input,
        gradouts(i))

      if(currentGradInput != null) {
        if(i == 0) {
          if (!concatCopy(this.gradInput, currentGradInput)) {
            println("STRIDE NOT EQUAL 1")
            this.gradInput.copy(currentGradInput)
          }
        } else {
          this.gradInput.add(currentGradInput)
        }
      }
      i += 1
      offset += currentOutput.size(dimension)
    }

    backwardTime += System.nanoTime() - before
    this.gradInput
  }

  // Todo: this is different from torch accUpdateGradParameters
  override def updateParameters(learningRate: T): Unit = {
    var offset = 1
    var i = 0
    while(i < this.modules.length) {
      val currentOutput = this.modules(i).output
      this.modules(i).updateParameters(learningRate)
      i += 1
      offset += currentOutput.size(dimension)
    }
  }

  override def equals(obj : Any) : Boolean = {
    if(!super.equals(obj)) {
      return false
    }

    if(!obj.isInstanceOf[Concat[T]])
      return false
    val other = obj.asInstanceOf[Concat[T]]
    if(this.eq(other))
      return true
    if(dimension != other.dimension)
      return false

    if(this.modules.length != other.modules.length)
      return false

    val moduleLength = modules.length
    var i = 0
    while(i < moduleLength) {
      if(modules(i) != other.modules(i)) {
        return false
      }
      i += 1
    }

    true
  }

  /**
   * Get the stride discard dimensions with size 1
   * @param tensor tensor
   * @return
   */
  def getStride[@specialized(Float, Double) T](tensor : Tensor[T]): Int = {
    var d = tensor.nDimension()
    while(d > 0) {
      if(tensor.size(d) != 1) {
        return tensor.stride(d)
      }
      d -= 1
    }

    0
  }

  def getLargestContiguousSize[@specialized(Float, Double) T](tensor : Tensor[T]) : (Int, Int) = {
    var largestSize = 1
    var largestDim = tensor.nDimension()
    while(largestDim > 0) {
      if(tensor.size(largestDim) != 1) {
        if(tensor.stride(largestDim) == largestSize) {
          largestSize = largestSize * tensor.size(largestDim)
        }
        else
          return (largestDim, largestSize)
      }
      largestDim -= 1
    }
    (largestDim, largestSize)
  }

  def getCounter(largestDim : Int) : Array[Int] = {
    val counter = new Array[Int](largestDim)
    var d = 0
    while (d < largestDim) {
      counter(d) = 0
      d += 1
    }
    counter
  }

  def updateCounter[@specialized(Float, Double) T](tensor : Tensor[T], counter : Array[Int], offset : Int, dim : Int) : (Boolean, Int) = {
    if(dim == 0) {
      return (true, offset)
    }

    var _offset = offset
    var i = dim
    while(i > 0) {
      counter(i - 1) += 1
      _offset += tensor.stride(i)
      if(counter(i - 1) == tensor.size(i)) {
        if(i == 1) {
          return (true, _offset)
        } else {
          _offset -= counter(i - 1) * tensor.stride(i)
          counter(i - 1) = 0
        }
      } else {
        return (false, _offset)
      }
      i -= 1
    }

    (false, _offset)
  }

  override def toString() : String = {
    val tab = "  "
    val next = "  |`-> "
    val last = "   ... -> "
    val ext = "  |    "
    val extlast = "       "
    s"nn.Concat {$line${tab}input$line${modules.zipWithIndex
      .map{case (model : Module[T], index : Int) => s"$tab$next(${index + 1}): ${
        if(index == modules.length - 1)
          model.setLine(line + tab + extlast)
        else
          model.setLine(line + tab + ext)
      }"}
      .mkString(line)}$line$tab${last}output$line$tab}"
  }
}