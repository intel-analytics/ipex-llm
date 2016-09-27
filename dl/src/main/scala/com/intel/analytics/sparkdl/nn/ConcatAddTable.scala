package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
  * Created by yao on 9/21/16.
  */
class ConcatAddTable[T: ClassTag](ip: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends Container[T] {
  val concatOutput = ArrayBuffer[Tensor[T]]()
  val cAddGradInput = ArrayBuffer[Tensor[T]]()

  override def updateOutput(input: Tensor[T]): Tensor[T] ={
    concatTableUpdateOutput(input)
    cADDTableUpdateOutput(concatOutput)
  }

  def concatTableUpdateOutput(input: Tensor[T]): ArrayBuffer[Tensor[T]] = {
    for ((module, i) <- modules.zipWithIndex) {
      updateBuffer(i, concatOutput, module.updateOutput(input))
    }

    concatOutput
  }

  def cADDTableUpdateOutput(input: ArrayBuffer[Tensor[T]]): Tensor[T] = {
    if (ip) {
      output.set(input(0))
    } else {
      output.resizeAs(input(0)).copy(input(0))
    }

    var i = 1
    while (i < input.length) {
      output.add(input(i))
      i += 1
    }

    output
  }


  def concatTableUpdateGradInput(input: Tensor[T], gradOutputs: ArrayBuffer[Tensor[T]]): Tensor[T] = {
    for ((module, i) <- modules.zipWithIndex) {
      val gradOutput = gradOutputs(i)
      val currentGradInput = module.updateGradInput(input, gradOutput)
      if (i == 0) {
        gradInput = gradInput.resizeAs(currentGradInput).copy(currentGradInput)
      } else {
        gradInput.add(currentGradInput)
      }
    }

    gradInput
  }

  def updateBuffer(i: Int, arrayBuffer: ArrayBuffer[Tensor[T]], tensor: Tensor[T]) = {
    if (i < arrayBuffer.size)
      arrayBuffer.update(i, tensor)
    else
      arrayBuffer.append(tensor)
  }

  def cAddTableUpdateGradInput(input: ArrayBuffer[Tensor[T]], gradOutputs: Tensor[T]): ArrayBuffer[Tensor[T]] = {
    for ((in, i) <- input.zipWithIndex) {
      if (ip)
        updateBuffer(i, cAddGradInput, gradOutputs)
      else
        updateBuffer(i, cAddGradInput, Tensor().resizeAs(in).copy(gradOutputs))
    }

    cAddGradInput
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    cAddTableUpdateGradInput(concatOutput, gradOutput)
    concatTableUpdateGradInput(input, cAddGradInput)
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double): Unit = {
    for ((module, i) <- modules.zipWithIndex) {
      module.accGradParameters(input, cAddGradInput(i), scale)
    }
  }

  override def toString: String = {
    val tab = "  "
    val line = "\n"
    val next = "  |`-> "
    val lastNext = "   `-> "
    val ext = "  |    "
    val extlast = "       "
    val last = "   ... -> "
    s"nn.Concat {$line${tab}input$line${
      modules.zipWithIndex
        .map { case (model: Module[T], index: Int) => {
          s"$tab$next(${index + 1}): ${
            if (index == modules.length - 1) {
              model.setLine(line + tab + extlast)
            } else {
              model.setLine(line + tab + ext)
            }
          }"
        }
             }.mkString(line)
    }$line$tab${last}output$line$tab}"
  }
}
