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

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * Concat concatenates the output of one layer of "parallel"
 * modules along the provided {@code dimension}: they take the
 * same inputs, and their output is concatenated.
 *                 +-----------+
 *            +---->  module1  -----+
 *            |    |           |    |
 * input -----+---->  module2  -----+----> output
 *            |    |           |    |
 *            +---->  module3  -----+
 *                 +-----------+
 *
 * @param dimension dimension
 */
@SerialVersionUID(- 5218461876031660707L)
class Concat[T: ClassTag](val dimension: Int)(
  implicit ev: TensorNumeric[T]) extends DynamicContainer[Tensor[T], Tensor[T], T] {

  private var size: Array[Int] = null
  @transient
  private var results: Array[Future[Unit]] = null
  @transient
  private var gradouts: Array[Tensor[T]] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val outs = new Array[Tensor[T]](this.modules.length)
    var i = 0
    while (i < this.modules.length) {
      val currentOutput = this.modules(i)
        .forward(input.asInstanceOf[Activity])
        .asInstanceOf[Tensor[T]]

      outs(i) = currentOutput.asInstanceOf[Tensor[T]]
      if (i == 0) {
        this.size = currentOutput.size()
      } else {
        require(this.size.length == currentOutput.size.length,
        s"${this.modules(i).getName} output size mismatch, expected : ${this.size.length}," +
          s"actual ${currentOutput.size.length}")
        var index = 0
        val ssize = this.size.length
        while (index < ssize) {
          if (index != dimension - 1) {
            require(this.size(index) == currentOutput.size(index + 1),
              s"${this.modules(i).getName} output size at dimension ${index + 1} mismatch," +
                s"expected ${this.size(index)}, actual : ${currentOutput.size(index + 1)}")
          }
          index += 1
        }
        this.size(this.dimension - 1) += currentOutput.size(this.dimension)
      }
      i += 1
    }
    this.output.resize(this.size)
    if (results == null || results.length != this.modules.length) {
      results = new Array[Future[Unit]](this.modules.length)
    }

    var offset = 1
    i = 0
    while (i < this.modules.length) {
      val currentOutput = outs(i)
      val _offset = offset
      results(i) = Engine.model.invoke(() => {
        val target = this.output.narrow(this.dimension, _offset,
          currentOutput.size(this.dimension))
        if (target.isContiguous() || this.dimension > 2) {
          // Copy directly when target is Contiguous or dimension is larger than 2
          // in which case the contiguous region in target tensor is fairly small in practice
          target.copy(currentOutput)
        } else {
          // Divide target into contiguous frames when target isn't contiguous
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
      offset += currentOutput.size(this.dimension)
    }

    Engine.model.sync(results)

    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    var before = System.nanoTime()
    this.gradInput.resizeAs(input)
    var offset = 1
    if (gradouts == null || gradouts.length != this.modules.length) {
      gradouts = new Array[Tensor[T]](this.modules.length)
    }
    var i = 0
    while (i < this.modules.length) {
      val currentOutput = this.modules(i).output.asInstanceOf[Tensor[T]]
      val _offset = offset
      val _i = i
      results(i) = Engine.model.invoke( () => {
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
      })
      i += 1
      offset += currentOutput.size(dimension)
    }
    Engine.model.sync(results)

    i = 0
    offset = 1
    while (i < this.modules.length) {
      val currentOutput = this.modules(i).output.asInstanceOf[Tensor[T]]
      val currentGradInput = this.modules(i)
        .updateGradInput(input.asInstanceOf[Activity], gradouts(i).asInstanceOf[Activity])
        .asInstanceOf[Tensor[T]]

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

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    var offset = 1
    var i = 0
    while (i < this.modules.length) {
      val currentOutput = this.modules(i).output.asInstanceOf[Tensor[T]]
      this.modules(i).accGradParameters(
        input.asInstanceOf[Activity],
        gradOutput.narrow(dimension, offset, currentOutput.size(dimension))
          .asInstanceOf[Activity])

      i += 1
      offset += currentOutput.size(dimension)
    }
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val before = System.nanoTime()
    this.gradInput.resizeAs(input)
    var offset = 1
    if (gradouts == null || gradouts.length != this.modules.length) {
      gradouts = new Array[Tensor[T]](this.modules.length)
    }
    var i = 0
    while (i < this.modules.length) {
      val currentOutput = this.modules(i).output.asInstanceOf[Tensor[T]]
      val _offset = offset
      val _i = i
      results(i) = Engine.model.invoke( () => {
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
      })
      i += 1
      offset += currentOutput.size(dimension)
    }
    Engine.model.sync(results)

    i = 0
    offset = 1
    while (i < this.modules.length) {
      val currentOutput = this.modules(i).output.asInstanceOf[Tensor[T]]
      val currentGradInput = this.modules(i)
        .backward(input.asInstanceOf[Activity], gradouts(i).asInstanceOf[Activity])
        .asInstanceOf[Tensor[T]]

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
    backwardTime += System.nanoTime() - before

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
    s"${getPrintName}{$line${tab}input$line${
      modules.zipWithIndex
        .map { case (model: AbstractModule[Activity, Activity, T], index: Int)
        => s"$tab$next(${index + 1}): ${
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
    forwardTime = 0
    backwardTime = 0
  }

  override def getEndNodes(startNodes: Array[ModuleNode[T]]): Array[ModuleNode[T]] = {
    val outputs = ArrayBuffer[ModuleNode[T]]()
    var outputTuple: Array[ModuleNode[T]] = null
    for (i <- 0 to modules.size - 1) {
      outputTuple = modules(i).getEndNodes(startNodes)
      outputs ++= outputTuple
    }
    Array(JoinTable(dimension, -1).inputs(outputs: _*))
  }
}

object Concat {
  def apply[@specialized(Float, Double) T: ClassTag](
      dimension: Int)(implicit ev: TensorNumeric[T]) : Concat[T] = {
    new Concat[T](dimension)
  }
}
