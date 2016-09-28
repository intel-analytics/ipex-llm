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

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag
import com.intel.analytics.sparkdl.utils.Engine

class Concat[T: ClassTag](val dimension: Int)(
  implicit ev: TensorNumeric[T]) extends Container[T] {
  private var size: Array[Int] = null
  @transient
  private var results: Array[Future[Unit]] = null
  private var gradouts: Array[Tensor[T]] = null

  protected var forwardTimeOverhead = 0L

  def getSize(): Array[Int] = {
    return size
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
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
    val before = System.nanoTime()
    this.output.resize(this.size)
    if (results == null || results.length != this.modules.length) {
      results = new Array[Future[Unit]](this.modules.length)
    }

    var offset = 1
    i = 0
    while (i < this.modules.length) {
      val currentOutput = outs(i)
      val _offset = offset
      results(i) = Future {
        val target = this.output.narrow(this.dimension, _offset,
          currentOutput.size(this.dimension))
        var f = 1
        while (f <= target.size(1)) {
          val curFrame = target.select(1, f)
          val outputFrame = currentOutput.select(1, f)
          require(curFrame.isContiguous())
          require(outputFrame.isContiguous())
          curFrame.copy(outputFrame)
          f += 1
        }
      }(Engine.getInstance())
      i += 1
      offset += currentOutput.size(this.dimension)
    }

    i = 0
    while (i < results.length) {
      Await.result(results(i), Duration.Inf)
      i += 1
    }
    forwardTimeOverhead += System.nanoTime() - before

    this.output
  }

  override def getTimes(): Array[(Module[T], Long, Long)] = {
    this.modules.map(_.getTimes()).flatten.toArray ++
      Array((this, forwardTimeOverhead, backwardTime))
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    this.gradInput.resizeAs(input)

    var offset = 1
    var i = 0
    while (i < this.modules.length) {
      val currentOutput = this.modules(i).output
      val currentGradInput = this.modules(i).updateGradInput(input,
        gradOutput.narrow(dimension, offset, currentOutput.size(dimension)))

      if (currentGradInput != null) {
        if (i == 0) {
          require(this.gradInput.isContiguous())
          require(currentGradInput.isContiguous())
          this.gradInput.copy(currentGradInput)
        } else {
          this.gradInput.add(currentGradInput)
        }
      }
      i += 1
      offset += currentOutput.size(dimension)
    }

    this.gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
    scale: Double = 1.0): Unit = {
    var offset = 1
    var i = 0
    while (i < this.modules.length) {
      val currentOutput = this.modules(i).output
      this.modules(i).accGradParameters(
        input,
        gradOutput.narrow(dimension, offset, currentOutput.size(dimension)),
        scale)

      i += 1
      offset += currentOutput.size(dimension)
    }
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    var before = System.nanoTime()
    this.gradInput.resizeAs(input)
    var offset = 1
    if (gradouts == null || gradouts.length != this.modules.length) {
      gradouts = new Array[Tensor[T]](this.modules.length)
    }
    var i = 0
    while (i < this.modules.length) {
      val currentOutput = this.modules(i).output
      val _offset = offset
      val _i = i
      results(i) = Future {
        val narrowedTensor = gradOutput.narrow(dimension, _offset,
          currentOutput.size(dimension))
        if(dimension == 2) {
          gradouts(_i) = Tensor[T]().resizeAs(narrowedTensor)
          var b = 1
          val firstSize = narrowedTensor.size(1)
          while(b <= firstSize) {
            gradouts(_i).select(1, b).copy(narrowedTensor.select(1, b))
            b += 1
          }
        } else {
          gradouts(_i) = narrowedTensor.contiguous()
        }
      }(Engine.getInstance())
      i += 1
      offset += currentOutput.size(dimension)
    }
    i = 0
    while (i < this.modules.length) {
      Await.result(results(i), Duration.Inf)
      i += 1
    }
    backwardTime += System.nanoTime() - before

    i = 0
    offset = 1
    while (i < this.modules.length) {
      val currentOutput = this.modules(i).output
      val currentGradInput = this.modules(i).backward(input,
        gradouts(i))

      before = System.nanoTime()
      if (currentGradInput != null) {
        if (i == 0) {
          require(this.gradInput.isContiguous())
          require(currentGradInput.isContiguous())
          this.gradInput.copy(currentGradInput)
        } else {
          this.gradInput.add(currentGradInput)
        }
      }
      i += 1
      offset += currentOutput.size(dimension)
      backwardTime += System.nanoTime() - before
    }

    this.gradInput
  }

  // Todo: this is different from torch accUpdateGradParameters
  override def updateParameters(learningRate: T): Unit = {
    var offset = 1
    var i = 0
    while (i < this.modules.length) {
      val currentOutput = this.modules(i).output
      this.modules(i).updateParameters(learningRate)
      i += 1
      offset += currentOutput.size(dimension)
    }
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

  override def hashCode() : Int = {

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
    s"nn.Concat {$line${tab}input$line${
      modules.zipWithIndex
        .map { case (model: Module[T], index: Int) => s"$tab$next(${index + 1}): ${
          if (index == modules.length - 1) {
            model.setLine(line + tab + extlast)
          } else {
            model.setLine(line + tab + ext)
          }
        }"
        }
        .mkString(line)
    }$line$tab${last}output$line$tab}"
  }

  override def resetTimes(): Unit = {
    forwardTimeOverhead = 0
    forwardTime = 0
    backwardTime = 0
  }
}
