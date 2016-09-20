package com.intel.analytics.sparkdl.nn.mkl

import com.intel.analytics.sparkdl.mkl.MKL
import com.intel.analytics.sparkdl.nn.Module
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.language.implicitConversions

import scala.reflect.ClassTag

class ReLU[@specialized(Float, Double) T: ClassTag](ip:Boolean = false)(implicit ev: TensorNumeric[T]) extends Module[T]{
  override def toString() : String = {
    s"mkl.ReLU"
  }

  private var firstPass = true
  var classPtr = 0L;

  override def getClassPtr(): Long = classPtr

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(gradOutput)
    // TODO Why does copy in mkl_dnn? Because it costs so much time, I comment is out.
    // gradInput.copy(gradOutput)

    val inputOffset      = input.storageOffset()      - 1;
    val outputOffset     = output.storageOffset()     - 1;
    val gradInputOffset  = gradInput.storageOffset()  - 1;
    val gradOutputOffset = gradOutput.storageOffset() - 1;

    // +---------+-------+-------+
    // |         | 3-dim | 4-dim |
    // +=========+=======+=======+
    // | Number  | ?     | 1     |
    // +---------+-------+-------+
    // | Channel | 1     | 2     |
    // +---------+-------+-------+
    // | Height  | 2     | 3     |
    // +---------+-------+-------+
    // | Width   | 3     | 4     |
    // +---------+-------+-------+
    // Table: Index of 3-dim/4-dim input

    val inputWidth   = input.size(input.dim())
    val inputHeight  = input.size(input.dim() - 1)
    val inputChannel = if (input.dim() <= 2) 1 else input.size(input.dim() - 2)
    val inputNumber  = if (input.dim() <= 3) 1 else input.size(input.dim() - 3)
    // TODO we may set input.size(input.dim() - 3) == 1 if input.dim() == 3

    implicit def bool2int(b:Boolean) = if (b) 1 else 0
    val start = System.nanoTime()
    ev.getType() match {
      case "Float" => MKL.ReLUBackwardFloat(
        input.storage().array().asInstanceOf[Array[Float]], inputOffset,
        gradOutput.storage().array().asInstanceOf[Array[Float]], gradOutputOffset,
        gradInput.storage().array().asInstanceOf[Array[Float]],  gradInputOffset, classPtr)

      case "Double" => MKL.ReLUBackwardDouble(
        input.storage().array().asInstanceOf[Array[Double]], inputOffset,
        gradOutput.storage().array().asInstanceOf[Array[Double]], gradOutputOffset,
        gradInput.storage().array().asInstanceOf[Array[Double]],  gradInputOffset, classPtr)

      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
    //println("[SCALA] ReLU backward call JNI " + (System.nanoTime() - start) / 1e6)

    gradInput
  }

  override  def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)

    val inputOffset  = input.storageOffset()  - 1;
    val outputOffset = output.storageOffset() - 1;

    // +---------+-------+-------+
    // |         | 3-dim | 4-dim |
    // +=========+=======+=======+
    // | Number  | ?     | 1     |
    // +---------+-------+-------+
    // | Channel | 1     | 2     |
    // +---------+-------+-------+
    // | Height  | 2     | 3     |
    // +---------+-------+-------+
    // | Width   | 3     | 4     |
    // +---------+-------+-------+
    // Table: Index of 3-dim/4-dim input

    val inputWidth   = input.size(input.dim())
    val inputHeight  = input.size(input.dim() - 1)
    val inputChannel = if (input.dim() <= 2) 1 else input.size(input.dim() - 2)
    val inputNumber  = if (input.dim() <= 3) 1 else input.size(input.dim() - 3)
    // TODO we may set input.size(input.dim() - 3) == 1 if input.dim() == 3


    if (firstPass) {
      ev.getType() match {
        case "Float"  => classPtr = MKL.ReLUInitFloat(
          inputNumber, inputChannel, inputHeight, inputWidth, 4);
        case "Double"  => classPtr = MKL.ReLUInitDouble(
          inputNumber, inputChannel, inputHeight, inputWidth, 4);
        case _   => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
      firstPass = false
    }

    implicit def bool2int(b:Boolean) = if (b) 1 else 0
    val start = System.nanoTime()
    ev.getType() match {
      case "Float" => MKL.ReLUForwardFloat(
        input.storage().array().asInstanceOf[Array[Float]], inputOffset,
        output.storage().array().asInstanceOf[Array[Float]], outputOffset, classPtr)

      case "Double" => MKL.ReLUForwardDouble(
        input.storage().array().asInstanceOf[Array[Double]], inputOffset,
        output.storage().array().asInstanceOf[Array[Double]], outputOffset, classPtr)

      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
    //println("[SCALA] ReLU forward call JNI " + (System.nanoTime() - start) / 1e6)

    output
  }
}
