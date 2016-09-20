package com.intel.analytics.sparkdl.nn.mkl

import com.intel.analytics.sparkdl.mkl.MKL
import com.intel.analytics.sparkdl.nn.Module
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor._
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import scala.reflect.ClassTag
import scala.language.implicitConversions

/**
  * Created by wyz on 16-9-7.
  */
class LocalNormalizationAcrossChannels[@specialized(Float, Double) T: ClassTag]
(val size : Int = 5, val alpha : Double = 1.0, val beta : Double = 0.75, val k : Double = 1.0)(
  implicit ev: TensorNumeric[T]) extends Module[T] {

  private val scale = Tensor[T]()
  private val paddedSquare = Tensor[T]()
  private val paddedRatio = Tensor[T]()
  private val accumRatio = Tensor[T]()
  private val accumRatioTimeInput = Tensor[T]()

  require(size % 2 == 1, "LRN only supports odd values for size")
  val prePad = (size - 1) / 2

  var classPtr = 0L
  private var firstPass = true

  override def getClassPtr(): Long = classPtr

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[LocalNormalizationAcrossChannels[T]])
      return false
    val other = obj.asInstanceOf[LocalNormalizationAcrossChannels[T]]
    if (this.eq(other))
      return true

    size == other.size &&
      alpha == other.alpha && beta == other.beta && k == other.k
  }

  override def toString(): String = {
    s"mkl.LocalResponseNormalizationAcrossChannels($size, $alpha, $beta, $k)"
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 4, "Input must have 4 dimensions, corresponding to (batch, channels, height, width)")
    require(input.isContiguous(), "Input is not contiguous")

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
    val inputChannel = if (input.dim() <= 3) 1 else input.size(input.dim() - 2)
    val inputNumber  = if (input.dim() <= 3) 1 else input.size(input.dim() - 3)
    // TODO we may set input.size(input.dim() - 3) == 1 if input.dim() == 3

    if (firstPass) {
      ev.getType() match {
        case "Float" => classPtr = MKL.LRNInitFloat(
          inputNumber, inputChannel, inputHeight, inputWidth,
          size, alpha.toFloat, beta.toFloat, k.toFloat, 4)
        case "Double" => classPtr = MKL.LRNInitDouble(
          inputNumber, inputChannel, inputHeight, inputWidth,
          size, alpha.toDouble, beta.toDouble, k.toDouble, 4)
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
      firstPass = false
    }

    implicit def bool2int(b:Boolean) = if (b) 1 else 0
    ev.getType() match {
      case "Float" => MKL.LRNForwardFloat(
        input.storage().array().asInstanceOf[Array[Float]], inputOffset,
        output.storage().array().asInstanceOf[Array[Float]], outputOffset,
        classPtr
      )
      case "Double" => MKL.LRNForwardDouble(
        input.storage().array().asInstanceOf[Array[Double]], inputOffset,
        output.storage().array().asInstanceOf[Array[Double]], outputOffset,
        classPtr
      )
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 4, "Input must have 4 dimensions, corresponding to (batch, channels, height, width)")
    require(gradOutput.isContiguous(), "gradOutput is not contiguous")

    gradInput.resizeAs(input)

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
    val inputChannel = input.size(input.dim() - 2)
    val inputNumber  = if (input.dim() == 3) 1 else input.size(input.dim() - 3)
    // TODO we may set input.size(input.dim() - 3) == 1 if input.dim() == 3

    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputOffset = gradInput.storageOffset() -1

    ev.getType() match {
      case "Float" =>  MKL.LRNBackwardFloat(
        input.storage().array().asInstanceOf[Array[Float]], inputOffset,
        gradOutput.storage().array().asInstanceOf[Array[Float]], gradOutputOffset,
        gradInput.storage().array().asInstanceOf[Array[Float]], gradInputOffset,
        classPtr)
      case "Double" =>  MKL.LRNBackwardDouble(
        input.storage().array().asInstanceOf[Array[Double]], inputOffset,
        gradOutput.storage().array().asInstanceOf[Array[Double]], gradOutputOffset,
        gradInput.storage().array().asInstanceOf[Array[Double]], gradInputOffset,
        classPtr)
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }

    gradInput
  }
}
