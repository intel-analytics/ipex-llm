package com.intel.analytics.sparkdl.nn.mkl

import com.intel.analytics.sparkdl.mkl.MKL
import com.intel.analytics.sparkdl.nn.Module
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.RandomGenerator
import com.intel.analytics.sparkdl.tensor.Tensor

import scala.language.implicitConversions

import scala.reflect.ClassTag

class SpatialPooling[@specialized(Float, Double) T: ClassTag](val kernelWidth: Int,
                                                                 val kernelHeight: Int,
                                                                 val strideWidth: Int,
                                                                 val strideHeight: Int,
                                                                 val padWidth: Int = 0,
                                                                 val padHeight: Int = 0)
                                                                (implicit ev: TensorNumeric[T]) extends Module[T] {
  implicit def bool2int(b: Boolean) = if (b) 1 else 0

  var classPtr: Long = 0L
  private var firstPass = true

  val algorithm = 0;

  override def getClassPtr(): Long = classPtr

  // TODO just for adopt to the testcase
  var ceil_mode = false
  def ceil(): SpatialPooling[T] = {
    ceil_mode = true
    this
  }

  def floor(): SpatialPooling[T] = {
    ceil_mode = false
    this
  }

  override def toString() : String = {
    s"mkl.Pooling"
  }

  def this(kernelWidth: Int, kernelHeight: Int)(implicit ev: TensorNumeric[T]){
    this(kernelWidth, kernelHeight, kernelWidth, kernelHeight)
  }

  // compute the output height and width
  def computeOut(input:Int, pad:Int, kernel:Int, stride:Int): Int = {
    if (ceil_mode)
      math.ceil(1.0 * (input + 2 * pad - kernel) / stride).toInt + 1
    else
      math.floor(1.0 * (input + 2 * pad - kernel) / stride).toInt + 1
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)

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
    val inputChannel = input.size(input.dim() - 2)
    val inputNumber  = if (input.dim() == 3) 1 else input.size(input.dim() - 3)
    // TODO we may set input.size(input.dim() - 3) == 1 if input.dim() == 3

    val outputHeight  = computeOut(inputHeight, padHeight, kernelHeight, strideHeight)
    val outputWidth   = computeOut(inputWidth,  padHeight, kernelWidth,  strideWidth)
    val outputChannel = inputChannel
    val outputNumber  = inputNumber

    ev.getType() match {
      case "Float" => MKL.PoolingBackwardFloat(
        input.storage().array().asInstanceOf[Array[Float]], inputOffset,
        gradOutput.storage().array().asInstanceOf[Array[Float]], gradOutputOffset,
        gradInput.storage().array().asInstanceOf[Array[Float]], gradInputOffset,
        classPtr)
      case "Double" => MKL.PoolingBackwardDouble(
        input.storage().array().asInstanceOf[Array[Double]], inputOffset,
        gradOutput.storage().array().asInstanceOf[Array[Double]], gradOutputOffset,
        gradInput.storage().array().asInstanceOf[Array[Double]], gradOutputOffset,
        classPtr)
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }

    gradInput
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
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
    val inputChannel = input.size(input.dim() - 2)
    val inputNumber  = if (input.dim() == 3) 1 else input.size(input.dim() - 3)
    // TODO we may set input.size(input.dim() - 3) == 1 if input.dim() == 3

    val outputHeight  = computeOut(inputHeight, padHeight, kernelHeight, strideHeight)
    val outputWidth   = computeOut(inputWidth,  padWidth,  kernelWidth,  strideWidth)
    val outputChannel = inputChannel
    val outputNumber  = inputNumber

    val inputOffset  = input.storageOffset()  - 1;
    val outputOffset = output.storageOffset() - 1;

    if (input.dim() == 3)
      output.resize(Array(outputChannel, outputHeight, outputWidth))
    else
      output.resize(Array(outputNumber, outputChannel, outputHeight, outputWidth))

    // TODO algorithm = 0 means using MAX
    val algorithm = 0

    if (firstPass) {
      ev.getType() match {
        case "Float"  => classPtr = MKL.PoolingInitFloat(
          inputNumber, inputChannel, inputHeight, inputWidth,
          kernelHeight, kernelWidth, strideHeight, strideWidth, padHeight, padWidth, 4,
          ceil_mode, algorithm)
        case "Double"  => classPtr = MKL.PoolingInitDouble(
          inputNumber, inputChannel, inputHeight, inputWidth,
          kernelHeight, kernelWidth, strideHeight, strideWidth, padHeight, padWidth, 4,
          ceil_mode, algorithm)
      case _   => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }

      firstPass = false
    }

    ev.getType() match {
      case "Float" => MKL.PoolingForwardFloat(
        input.storage().array.asInstanceOf[Array[Float]], inputOffset,
        output.storage().array.asInstanceOf[Array[Float]], outputOffset, classPtr)
      case "Double" => MKL.PoolingForwardDouble(
        input.storage().array.asInstanceOf[Array[Double]], inputOffset,
        output.storage().array.asInstanceOf[Array[Double]], outputOffset, classPtr)
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
    output
  }
}

class SpatialMaxPooling[T: ClassTag](kernelWidth: Int,
                                     kernelHeight: Int,
                                     strideWidth : Int,
                                     strideHeight: Int,
                                     padWidth: Int = 0,
                                     padHeight: Int = 0)
                                    (implicit ev: TensorNumeric[T])
  extends SpatialPooling[T](kernelWidth, kernelHeight, strideWidth, strideHeight, padWidth, padHeight)
{
  override val algorithm: Int = 0
  def this(kernelWidth: Int, kernelHeight: Int)(implicit ev: TensorNumeric[T]){
    this(kernelWidth, kernelHeight, kernelWidth, kernelHeight)
  }
  override def toString() : String = {
    s"mkl.SpatialMaxPooling"
  }
}

class SpatialAveragePooling[T: ClassTag](kernelWidth: Int,
                                                                     kernelHeight: Int,
                                                                     strideWidth: Int,
                                                                     strideHeight: Int,
                                                                     padWidth: Int = 0,
                                                                     padHeight: Int = 0)
                                                                    (implicit ev: TensorNumeric[T])
  extends SpatialPooling[T](kernelWidth, kernelHeight, strideWidth, strideHeight, padWidth, padHeight)
{
  override val algorithm: Int = 1
  def this(kernelWidth: Int, kernelHeight: Int)(implicit ev: TensorNumeric[T]){
    this(kernelWidth, kernelHeight, kernelWidth, kernelHeight)
  }
  override def toString() : String = {
    s"mkl.SpatialAvgPooling"
  }
}
