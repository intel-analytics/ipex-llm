package com.intel.webscaleml.nn.nn.mkl_dnn

import com.intel.webscaleml.nn.mkl.Primitives
import com.intel.webscaleml.nn.nn.Module
import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric
import com.intel.webscaleml.nn.tensor.{torch, Tensor}
import com.intel.webscaleml.nn.tensor.RandomGenerator._

import scala.reflect.ClassTag

class SpatialConvolution[@specialized(Float, Double) T:ClassTag] (
                                                                   val nInputPlane : Int, // The number of expected input planes in the image given into forward()
                                                                   val nOutputPlane : Int,  // The number of output planes the convolution layer will produce.
                                                                   val kW : Int,  // The kernel width of the convolution
                                                                   val kH : Int,  // The kernel height of the convolution
                                                                   val dW : Int = 1,  // The step of the convolution in the width dimension.
                                                                   val dH : Int = 1,  //The step of the convolution in the height dimension
                                                                   val padW : Int = 0,  // The additional zeros added per width to the input planes. A good number is (kW-1)/2.
                                                                   val padH : Int = 0 // The additional zeros added per height to the input planes. A good number is (kH-1)/2.
                                                                   )(implicit ev: TensorNumeric[T]) extends Module[T] {

  val weight : Tensor[T] = torch.Tensor[T](nOutputPlane, nInputPlane, kH, kW)
  val bias : Tensor[T] = torch.Tensor[T](nOutputPlane)
  this.gradWeight = torch.Tensor[T](nOutputPlane, nInputPlane, kH, kW)
  this.gradBias = torch.Tensor[T](nOutputPlane)
  val fInput = torch.Tensor[T]()
  val fGradInput = torch.Tensor[T]()
  reset()

  override def reset(): Unit ={
    val stdv = 1.0 /math.sqrt(kW * kH * nInputPlane)
    weight.apply1(_=>ev.fromType[Double](RNG.uniform(0,1)*2*stdv - stdv)) //todo, better to support uniform
    bias.apply1(_=>ev.fromType[Double](RNG.uniform(0,1)*2*stdv - stdv))
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4, "Only support 3D or 4D(batch mode) input")
    require(input.isContiguous(), "input is not contiguous")

    val dimFrame = if(input.dim() == 3) 1 else 2
    val dimWidth = if(input.dim() == 3) 3 else 4
    val dimHeight = if(input.dim() == 3) 2 else 3
    val inputWidth = input.size(dimWidth)
    val inputHeight = input.size(dimHeight)

    val outputWidth = (inputWidth + 2*padW - kW) / dW + 1
    val outputHeight = (inputHeight + 2*padH - kH) / dH + 1

    require(outputWidth >= 1 && outputHeight >= 1, "output size is too small")
    if(input.dim() == 3) {
      output.resize(Array(nOutputPlane, outputHeight, outputWidth))
      ev.getType() match {
        case "Float" => Primitives.convolution_forward(
          input.storage().array().asInstanceOf[Array[Float]], input.storageOffset() - 1, 1, input.size(1), input.size(2), input.size(3),
          output.storage().array().asInstanceOf[Array[Float]], output.storageOffset() - 1, 1, output.size(1), output.size(2), output.size(3),
          weight.storage().array().asInstanceOf[Array[Float]], weight.storageOffset() - 1,
          bias.storage().array().asInstanceOf[Array[Float]], bias.storageOffset() - 1,
          1, 1, dH, dW,
          kH, kW, nOutputPlane, nInputPlane)

        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    } else {
      val batchSize = input.size(1)
      output.resize(Array(batchSize, nOutputPlane, outputHeight, outputWidth))
      ev.getType() match {
        case "Float" => Primitives.convolution_forward(
          input.storage().array().asInstanceOf[Array[Float]], input.storageOffset() - 1, input.size(1), input.size(2), input.size(3), input.size(4),
          output.storage().array().asInstanceOf[Array[Float]], output.storageOffset() - 1, output.size(1), output.size(2), output.size(3), output.size(4),
          weight.storage().array().asInstanceOf[Array[Float]], weight.storageOffset() - 1,
          bias.storage().array().asInstanceOf[Array[Float]], bias.storageOffset() - 1,
          1, 1, dH, dW,
          kH, kW, nOutputPlane, nInputPlane)

        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    }

    output
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]) : Tensor[T] = {
    require(input.nDimension() == 3 || input.nDimension() == 4, "Only support 3D or 4D input")
    require(nOutputPlane == (if(input.nDimension() == 3) gradOutput.size(1) else gradOutput.size(2)),
      "Number of output features is not equal to nOutputPlane")
    require(input.isContiguous(), "input is not contiguous")
    require(gradInput.isContiguous(), "gradInput is not contiguous")

    gradInput.resizeAs(input)

    if(input.dim() == 3) {
      ev.getType() match {
        case "Float" => Primitives.convolution_backward(
        gradInput.storage().array().asInstanceOf[Array[Float]], gradInput.storageOffset() - 1, 1, gradInput.size(1), gradInput.size(2), gradInput.size(3),
        gradWeight.storage().array().asInstanceOf[Array[Float]], gradWeight.storageOffset() - 1, gradBias.storage().array().asInstanceOf[Array[Float]], gradBias.storageOffset() - 1,
        gradOutput.storage().array().asInstanceOf[Array[Float]], gradOutput.storageOffset() - 1, 1, gradOutput.size(1), gradOutput.size(2), gradOutput.size(3),
        input.storage().array().asInstanceOf[Array[Float]], input.storageOffset() - 1, weight.storage().array().asInstanceOf[Array[Float]], weight.storageOffset() - 1,
        bias.storage().array().asInstanceOf[Array[Float]], bias.storageOffset() - 1, nOutputPlane,
        1, 1, dH, dW, nOutputPlane, nInputPlane, kH, kW)

        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    } else {
      ev.getType() match {
        case "Float" => Primitives.convolution_backward(
        gradInput.storage().array().asInstanceOf[Array[Float]], gradInput.storageOffset() - 1, gradInput.size(1), gradInput.size(2), gradInput.size(3), gradInput.size(4),
        gradWeight.storage().array().asInstanceOf[Array[Float]], gradWeight.storageOffset() - 1, gradBias.storage().array().asInstanceOf[Array[Float]], gradBias.storageOffset() - 1,
        gradOutput.storage().array().asInstanceOf[Array[Float]], gradOutput.storageOffset() - 1, gradOutput.size(1), gradOutput.size(2), gradOutput.size(3), gradOutput.size(4),
        input.storage().array().asInstanceOf[Array[Float]], input.storageOffset() - 1, weight.storage().array().asInstanceOf[Array[Float]], weight.storageOffset() - 1,
        bias.storage().array().asInstanceOf[Array[Float]], bias.storageOffset() - 1, nOutputPlane,
        1, 1, dH, dW, nOutputPlane, nInputPlane, kH, kW)

        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    }

    gradInput
  }

  override def updateParameters(learningRate:T): Unit ={
    weight.map(gradWeight,(a,b)=>ev.minus(a, ev.times(learningRate,b)))
    bias.map(gradBias,(a,b)=>ev.minus(a, ev.times(learningRate,b)))
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    gradBias.zero()
  }

  override  def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) ={
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def equals(obj : Any) : Boolean = {

    if(!super.equals(obj)) {
      return false
    }

    if(!obj.isInstanceOf[SpatialConvolution[T]])
      return false
    val other = obj.asInstanceOf[SpatialConvolution[T]]
    if(this.eq(other))
      return true

    nInputPlane == other.nInputPlane &&
      nOutputPlane == other.nOutputPlane &&
      kW == other.kW &&
      kH == other.kH &&
      dW == other.dW &&
      dH == other.dH &&
      padW == other.padW &&
      padH == other.padH &&
      weight == other.weight &&
      bias == other.bias &&
      gradWeight == other.gradWeight &&
      gradBias == other.gradBias
  }

  override def toString() : String = {
    s"mkl_dnn.SpatialConvolution($nInputPlane -> $nOutputPlane, $kW x $kH, $dW, $dH, $padW, $padH)"
  }

  override def findModel(paramOffset : Int, indexes : Array[Int]) : (Module[T], Int, Array[Int]) = {
    (this, paramOffset - nOutputPlane * nInputPlane * kH * kW - nOutputPlane, indexes)
  }

  /*mkl-dnn's convolution_backward has done updateGradInput and accGradParameters, so accGradParameters does nothing
   *
    */
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    backward(input, gradOutput)
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double = 1.0): Unit = {}
}
