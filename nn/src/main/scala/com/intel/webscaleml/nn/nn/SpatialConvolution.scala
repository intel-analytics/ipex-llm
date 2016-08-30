package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric
import com.intel.webscaleml.nn.tensor.TensorType.{FloatType, DoubleType}
import com.intel.webscaleml.nn.tensor._
import com.intel.webscaleml.nn.tensor.RandomGenerator._

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration.Duration
import scala.reflect.ClassTag
import scala.concurrent.{Await, ExecutionContext, Future}

class SpatialConvolution[@specialized(Float, Double) T:ClassTag] (
    val nInputPlane : Int, // The number of expected input planes in the image given into forward()
    val nOutputPlane : Int,  // The number of output planes the convolution layer will produce.
    val kW : Int,  // The kernel width of the convolution
    val kH : Int,  // The kernel height of the convolution
    val dW : Int = 1,  // The step of the convolution in the width dimension.
    val dH : Int = 1,  //The step of the convolution in the height dimension
    val padW : Int = 0,  // The additional zeros added per width to the input planes. A good number is (kW-1)/2.
    val padH : Int = 0, // The additional zeros added per height to the input planes. A good number is (kH-1)/2.
    private var initMethod : InitializationMethod = Default
  )(implicit ev: TensorNumeric[T]) extends Module[T] {

  val weight : Tensor[T] = torch.Tensor[T](nOutputPlane, nInputPlane, kH, kW)
  private var weightMM : Tensor[T] = null
  private var gradientBiasMT : Tensor[T] = null
  val bias : Tensor[T] = torch.Tensor[T](nOutputPlane)
  this.gradWeight = torch.Tensor[T](nOutputPlane, nInputPlane, kH, kW)
  private var gradWeightMM : Tensor[T] = null
  this.gradBias = torch.Tensor[T](nOutputPlane)
  val fInput = torch.Tensor[T]()
  val fGradInput = torch.Tensor[T]()
  private val ones = torch.Tensor[T]()
  private val onesBatch = torch.Tensor[T]()
  private val onesBias = torch.Tensor[T]()
  reset()

  private var im2colTime = 0L
  private var col2imTime = 0L

  def getIm2ColTime() = im2colTime
  def getCol2ImgTime() = col2imTime

  def setInitMethod(initMethod : InitializationMethod) : this.type = {
    this.initMethod = initMethod
    this
  }

  @transient
  private var results : Array[Future[Unit]] = null

  override def reset(): Unit ={
    initMethod match {
      case Default =>
        val stdv = 1.0 /math.sqrt(kW * kH * nInputPlane)
        weight.apply1(_=>ev.fromType[Double](RNG.uniform(0,1)*2*stdv - stdv)) //todo, better to support uniform
        bias.apply1(_=>ev.fromType[Double](RNG.uniform(0,1)*2*stdv - stdv))
      case Xavier =>
        val fanIn = nInputPlane * kH * kW
        val fanOut = nOutputPlane * kH * kW
        val stdv = math.sqrt(3 / (fanIn + fanOut))
        weight.apply1(_=>ev.fromType[Double](RNG.uniform(0,1)*2*stdv - stdv)) //todo, better to support uniform
        bias.fill(ev.fromType(0))
      case _ => ???
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4, "Only support 3D or 4D(batch mode) input")
    require(input.isContiguous())

    if(weightMM == null) {
      weightMM = weight.view(nOutputPlane, nInputPlane * kH * kW)
    }
    val dimWidth = if(input.dim() == 3) 3 else 4
    val dimHeight = if(input.dim() == 3) 2 else 3

    val inputWidth = input.size(dimWidth)
    val inputHeight = input.size(dimHeight)

    val outputWidth = (inputWidth + 2*padW - kW) / dW + 1
    val outputHeight = (inputHeight + 2*padH - kH) / dH + 1

    if(onesBias.dim() != 1 || onesBias.size(1) != outputHeight*outputWidth) {
      onesBias.resize(Array(outputHeight*outputWidth)).fill(ev.fromType(1.0))
    }

    require(outputWidth >= 1 && outputHeight >= 1, "output size is too small")
    if(input.dim() == 3) {
      require(input.size(1) == nInputPlane)
      require(input.isContiguous())
      val contiguousInput = input.contiguous()
      output.resize(Array(nOutputPlane, outputHeight, outputWidth))
      fInput.resize(Array(kW * kH * nInputPlane, outputHeight * outputWidth))
      updateOutputFrame(contiguousInput, output, weightMM, bias, fInput, kW, kH, dW, dH, padW, padH, nInputPlane,
        inputWidth, inputHeight, nOutputPlane, outputWidth, outputHeight)
    } else {
      require(input.size(2) == nInputPlane)
      val batchSize = input.size(1)
      output.resize(Array(batchSize, nOutputPlane, outputHeight, outputWidth))
      fInput.resize(Array(batchSize, kW * kH * nInputPlane, outputHeight * outputWidth))

      if(results == null || results.length != batchSize) {
        results = new Array[Future[Unit]](batchSize)
      }

      var i = 0
      while(i < batchSize) {
        val _i = i + 1
        results(i) = Future {
          val inputT = input.select(1, _i).contiguous()
          val outputT = output.select(1, _i)
          val fInputT = fInput.select(1, _i)
          updateOutputFrame(inputT, outputT, weightMM, bias, fInputT, kW, kH, dW, dH, padW, padH, nInputPlane,
            inputWidth, inputHeight, nOutputPlane, outputWidth, outputHeight)
        }
        i += 1
      }

      i = 0
      while(i < results.length) {
        Await.result(results(i), Duration.Inf)
        i += 1
      }
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 3 || input.nDimension() == 4, "Only support 3D or 4D input")
    gradInput.resizeAs(input)
    fGradInput.resizeAs(fInput)

    if(input.nDimension() == 3) {
      require(gradOutput.isContiguous())
      val contiguousGradOutput = gradOutput.contiguous()
      updateGradInputFrame(gradInput, contiguousGradOutput, weightMM.transpose(1, 2), fGradInput, kW, kH, dW, dH, padW, padH)
    } else {
      val transposedWeightMM = weightMM.transpose(1, 2)
      var i = 0
      val batchSize = input.size(1)
      while(i < batchSize) {
        val _i = i + 1
        results(i) = Future {
          val gradInputT = gradInput.select(1, _i)
          val gradOutputT = gradOutput.select(1, _i).contiguous()
          val fgradInputT = fGradInput.select(1, _i)
          updateGradInputFrame(gradInputT, gradOutputT, transposedWeightMM, fgradInputT, kW, kH, dW, dH, padW, padH)
        }
        i += 1
      }

      i = 0
      while(i < results.length) {
        Await.result(results(i), Duration.Inf)
        i += 1
      }
    }

    return gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double = 1.0): Unit = {
    require(input.nDimension() == 3 || input.nDimension() == 4, "Only support 3D or 4D input")
    val contiguousGradOutput = gradOutput.contiguous()

    if(input.nDimension() == 3) {
      if(gradWeightMM == null) {
        gradWeightMM = gradWeight.view(nOutputPlane, nInputPlane * kH * kW)
      }
      accGradParametersFrame(contiguousGradOutput, gradWeightMM, gradBias, fInput, ev.fromType[Double](scale))
    } else {
      val batchSize = input.size(1)
      if (gradWeightMM == null) {
        gradWeightMM = torch.Tensor[T]().resize(Array(batchSize, nOutputPlane, nInputPlane * kH * kW))
        gradientBiasMT = torch.Tensor[T]().resize(Array(batchSize, nOutputPlane))
      }
      if(ones.dim() != 1 || ones.size(1) != gradOutput.size(3) * gradOutput.size(4)) {
        ones.resize(Array(gradOutput.size(3) * gradOutput.size(4))).fill(ev.fromType(1.0))
      }

      if(onesBatch.dim() != 1 || onesBatch.size(1) != batchSize) {
        onesBatch.resize(Array(batchSize)).fill(ev.fromType(1.0))
      }
      var i = 0
      while(i < batchSize) {
        val _i = i + 1
        results(i) = Future {
          val gradOutputT = contiguousGradOutput.select(1, _i)
          val fInputT = fInput.select(1, _i)
          calcGradParametersFrame(gradOutputT, gradWeightMM.select(1, _i), gradientBiasMT.select(1, _i),
            fInputT, ev.fromType[Double](scale))
        }
        i += 1
      }

      i = 0
      while(i < results.length) {
        Await.result(results(i), Duration.Inf)
        i += 1
      }

      val gradView = gradWeightMM.view(batchSize, nOutputPlane * nInputPlane * kH * kW).t
      val grad = gradWeight.view(nOutputPlane * nInputPlane * kH * kW)
      grad.addmv(ev.fromType(1.0), ev.fromType(1.0), gradView, onesBatch)
      gradBias.addmv(ev.fromType(1.0), ev.fromType(1.0), gradientBiasMT.t, onesBatch)
    }
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
    s"nn.SpatialConvolution($nInputPlane -> $nOutputPlane, $kW x $kH, $dW, $dH, $padW, $padH)"
  }

  override def findModel(paramOffset : Int, indexes : Array[Int]) : (Module[T], Int, Array[Int]) = {
    (this, paramOffset - nOutputPlane * nInputPlane * kH * kW - nOutputPlane, indexes)
  }

  private def updateOutputFrame(input : Tensor[T], output : Tensor[T], weight : Tensor[T], bias : Tensor[T], fInput : Tensor[T],
     kW : Int, kH : Int, dW : Int, dH : Int, padW : Int, padH : Int,
     nInputPlane : Int, inputWidth : Int, inputHeight : Int,
     nOutputPlane : Int, outputWidth : Int, outputHeight : Int)(implicit ev: TensorNumeric[T]) : Unit = {

    val output2d = output.view(nOutputPlane, outputHeight*outputWidth)
    ev.getType() match {
      case "Double" => {
        val before = System.nanoTime()
        NNPrimitive.im2col_double(fInput.asInstanceOf[Tensor[Double]], input.asInstanceOf[Tensor[Double]], kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight)
        im2colTime += System.nanoTime() - before

        /*val output2d = torch.Tensor(output.storage(), output.storageOffset(), Array(nOutputPlane, outputHeight*outputWidth))
        var i = 0
        val biasIndex = Array(0)
        while(i < nOutputPlane) {
          biasIndex(0) = i + 1
          output2d.select(1, i + 1).fill(bias(biasIndex))
          i += 1
        }

        output2d.addmm(ev.fromType[Int](1), output2d, ev.fromType[Int](1), weight, fInput)*/
      }
      case "Float" => {
        val before = System.nanoTime()
        NNPrimitive.im2col_float(fInput.asInstanceOf[Tensor[Float]], input.asInstanceOf[Tensor[Float]], kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight)
        im2colTime += System.nanoTime() - before
        /*val output2d = torch.Tensor(output.storage(), output.storageOffset(), Array(nOutputPlane, outputHeight*outputWidth))
        var i = 0
        val biasIndex = Array(0)
        while(i < nOutputPlane) {
          biasIndex(0) = i + 1
          output2d.select(1, i + 1).fill(bias(biasIndex))
          i += 1
        }

        output2d.addmm(ev.fromType[Int](1), output2d, ev.fromType[Int](1), weight, fInput)*/

      }
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
    output2d.addmm(ev.fromType[Int](0), output2d, ev.fromType[Int](1), weight, fInput)
    output2d.addr(ev.fromType(1), bias, onesBias)
  }

  private def updateGradInputFrame(gradInput : Tensor[T], gradOutput : Tensor[T], weight : Tensor[T], fgradInput : Tensor[T],
      kW : Int, kH : Int, dW : Int, dH : Int, padW : Int, padH : Int)(implicit ev: TensorNumeric[T]) : Unit = {

    ev.getType() match {
      case "Double" => {
        val gradOutput2d = torch.Tensor(gradOutput.storage().asInstanceOf[Storage[Double]], gradOutput.storageOffset(),
          Array(gradOutput.size(1), gradOutput.size(2) * gradOutput.size(3)))
        fgradInput.asInstanceOf[Tensor[Double]].addmm(0.0, fgradInput.asInstanceOf[Tensor[Double]], 1.0, weight.asInstanceOf[Tensor[Double]], gradOutput2d)
        gradInput.asInstanceOf[Tensor[Double]].zero()
        val before = System.nanoTime()
        NNPrimitive.col2im_double(fgradInput.asInstanceOf[Tensor[Double]], gradInput.asInstanceOf[Tensor[Double]], kW, kH, dW, dH, padW, padH, gradInput.size(1), gradInput.size(3),
          gradInput.size(2), gradOutput.size(3), gradOutput.size(2))
        col2imTime += System.nanoTime() - before
      }
      case "Float" => {
        val gradOutput2d = torch.Tensor(gradOutput.storage().asInstanceOf[Storage[Float]], gradOutput.storageOffset(),
          Array(gradOutput.size(1), gradOutput.size(2) * gradOutput.size(3)))
        fgradInput.asInstanceOf[Tensor[Float]].addmm(0.0f, fgradInput.asInstanceOf[Tensor[Float]], 1.0f, weight.asInstanceOf[Tensor[Float]], gradOutput2d)
        gradInput.asInstanceOf[Tensor[Float]].zero()
        val before = System.nanoTime()
        NNPrimitive.col2im_float(fgradInput.asInstanceOf[Tensor[Float]], gradInput.asInstanceOf[Tensor[Float]], kW, kH, dW, dH, padW, padH, gradInput.size(1), gradInput.size(3),
          gradInput.size(2), gradOutput.size(3), gradOutput.size(2))
        col2imTime += System.nanoTime() - before
      }
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
  }

  private def accGradParametersFrame(gradOutput : Tensor[T], gradWeight : Tensor[T], gradBias : Tensor[T],
    fInput : Tensor[T], scale : T)(implicit ev: TensorNumeric[T]) : Unit = {

    ev.getType() match {
      case "Double" => {
        val gradOutput2d = torch.Tensor[Double](gradOutput.storage().asInstanceOf[Storage[Double]], gradOutput.storageOffset(),
          Array(gradOutput.size(1), gradOutput.size(2) * gradOutput.size(3)))

        gradWeight.asInstanceOf[Tensor[Double]].addmm(1.0, gradWeight.asInstanceOf[Tensor[Double]], ev.toType[Double](scale), gradOutput2d,
          fInput.t.asInstanceOf[Tensor[Double]])

        var i = 0
        while(i < gradBias.size(1)) {
          var sum = 0.0
          val data = gradOutput2d.storage().array()
          val offset = gradOutput2d.storageOffset() - 1 + i * gradOutput2d.stride(1)
          var k = 0
          while(k < gradOutput2d.size(2)) {
            sum += data(k + offset)
            k += 1
          }
          gradBias.asInstanceOf[Tensor[Double]].setValue(
            i + 1, gradBias.asInstanceOf[Tensor[Double]].valueAt(i + 1) + (ev.toType[Double](scale)*sum))
          i += 1
        }
      }
      case "Float" => {
        val gradOutput2d = torch.Tensor[Float](gradOutput.storage().asInstanceOf[Storage[Float]], gradOutput.storageOffset(),
          Array(gradOutput.size(1), gradOutput.size(2) * gradOutput.size(3)))

        gradWeight.asInstanceOf[Tensor[Float]].addmm(1.0f, gradWeight.asInstanceOf[Tensor[Float]], ev.toType[Float](scale), gradOutput2d,
          fInput.t.asInstanceOf[Tensor[Float]])

        var i = 0
        while(i < gradBias.size(1)) {
          var sum = 0.0f
          val data = gradOutput2d.storage().array()
          val offset = gradOutput2d.storageOffset() - 1 + i * gradOutput2d.stride(1)
          var k = 0
          while(k < gradOutput2d.size(2)) {
            sum += data(k + offset)
            k += 1
          }
          gradBias.asInstanceOf[Tensor[Float]].setValue(
            i + 1, gradBias.asInstanceOf[Tensor[Float]].valueAt(i + 1) + (ev.toType[Float](scale)*sum))
          i += 1
        }
      }
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
  }

  private def calcGradParametersFrame(gradOutput : Tensor[T], gradWeight : Tensor[T], gradBias : Tensor[T],
      fInput : Tensor[T], scale : T)(implicit ev: TensorNumeric[T]) : Unit = {

    ev.getType() match {
      case "Double" => {
        val gradOutput2d = torch.Tensor[Double](gradOutput.storage().asInstanceOf[Storage[Double]], gradOutput.storageOffset(),
          Array(gradOutput.size(1), gradOutput.size(2) * gradOutput.size(3)))

        gradWeight.asInstanceOf[Tensor[Double]].addmm(0.0, gradWeight.asInstanceOf[Tensor[Double]], ev.toType[Double](scale), gradOutput2d,
          fInput.t.asInstanceOf[Tensor[Double]])
        gradBias.asInstanceOf[Tensor[Double]].addmv(0.0, 1.0, gradOutput2d, ones.asInstanceOf[Tensor[Double]])
      }
      case "Float" => {
        val gradOutput2d = torch.Tensor[Float](gradOutput.storage().asInstanceOf[Storage[Float]], gradOutput.storageOffset(),
          Array(gradOutput.size(1), gradOutput.size(2) * gradOutput.size(3)))

        gradWeight.asInstanceOf[Tensor[Float]].addmm(0.0f, gradWeight.asInstanceOf[Tensor[Float]], ev.toType[Float](scale), gradOutput2d,
          fInput.t.asInstanceOf[Tensor[Float]])

        gradBias.asInstanceOf[Tensor[Float]].addmv(0.0f, 1.0f, gradOutput2d, ones.asInstanceOf[Tensor[Float]])
      }
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
  }
}
