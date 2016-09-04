package com.intel.analytics.dllib.lib.nn

import com.intel.analytics.dllib.lib.tensor.Tensor
import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag

class Concat[@specialized(Float, Double) T: ClassTag] (val dimension : Int)(implicit ev: TensorNumeric[T]) extends Container[T]{
  private var size : Array[Int] = null
  private var results : Array[Future[Unit]] = null
  private var gradouts : Array[Tensor[T]] = null

  def getSize(): Array[Int] ={
    return size
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
      results = new Array[Future[Unit]](this.modules.length)
    }

    var offset = 1
    i = 0
    while(i < this.modules.length) {
      val currentOutput = outs(i)
      val _offset = offset
      results(i) = Future {
        val target = this.output.narrow(this.dimension, _offset, currentOutput.size(this.dimension))//.copy(currentOutput)
        var f = 1
        while(f <= target.size(1)) {
          val curFrame = target.select(1, f)
          val outputFrame = currentOutput.select(1, f)
          require(curFrame.isContiguous())
          require(outputFrame.isContiguous())
          curFrame.copy(outputFrame)
          f += 1
        }
      }
      i += 1
      offset += currentOutput.size(this.dimension)
    }

    i = 0
    while(i < results.length) {
      Await.result(results(i), Duration.Inf)
      i += 1
    }

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

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
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
        gradouts(_i) = gradOutput.narrow(dimension, _offset, currentOutput.size(dimension)).contiguous()
      }
      i += 1
      offset += currentOutput.size(dimension)
    }
    i = 0
    while(i < this.modules.length) {
      Await.result(results(i), Duration.Inf)
      i += 1
    }

    i = 0
    offset = 1
    while(i < this.modules.length) {
      val currentOutput = this.modules(i).output
      val currentGradInput = this.modules(i).backward(input,
        gradouts(i))

      if(currentGradInput != null) {
        if(i == 0) {
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
