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

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor._
import com.intel.analytics.sparkdl.utils.{Engine, RandomGenerator}
import com.intel.analytics.sparkdl.utils.RandomGenerator._

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag

class SpatialConvolution[@specialized(Float, Double) T: ClassTag](
  val nInputPlane: Int, // The number of expected input planes in the image given into forward()
  val nOutputPlane: Int, // The number of output planes the convolution layer will produce.
  val kW: Int, // The kernel width of the convolution
  val kH: Int, // The kernel height of the convolution
  val dW: Int = 1, // The step of the convolution in the width dimension.
  val dH: Int = 1, // The step of the convolution in the height dimension
  val padW: Int = 0, // The additional zeros added per width to the input planes.
  val padH: Int = 0, // The additional zeros added per height to the input planes.
  val nGroup : Int = 1, // Kernel group number
  private var initMethod: InitializationMethod = Default
)(implicit ev: TensorNumeric[T]) extends Module[T] {

  require(nInputPlane % nGroup == 0, "Number of input channels should be multiples of group.")
  require(nOutputPlane % nGroup == 0, "Number of output channels should be multiples of group.")

  val weight: Tensor[T] = Tensor[T](nGroup, nOutputPlane / nGroup,
    nInputPlane / nGroup, kH, kW)
  this.gradWeight = Tensor[T](nGroup, nOutputPlane / nGroup, nInputPlane / nGroup, kH, kW)

  private var weightMM: Tensor[T] = null
  private var gradientBiasMT: Tensor[T] = null
  val bias: Tensor[T] = Tensor[T](nOutputPlane)
  //private var gradWeightMM: Tensor[T] = null
  var gradWeightMM: Tensor[T] = null
  this.gradBias = Tensor[T](nOutputPlane)
  var fInput = Tensor[T]()
  var fGradInput = Tensor[T]()
  private val ones = Tensor[T]()
  private val onesBatch = Tensor[T]()
  private val onesBias = Tensor[T]()
  reset()
  
  private var im2colTime = 0L
  private var col2imTime = 0L

  def getIm2ColTime(): Double = im2colTime

  def getCol2ImgTime(): Double = col2imTime

  def setInitMethod(initMethod: InitializationMethod): this.type = {
    this.initMethod = initMethod
    this
  }

  @transient
  private var results: Array[Future[Unit]] = null

  override def reset(): Unit = {
    initMethod match {
      case Default =>
        val stdv = 1.0 / math.sqrt(kW * kH * nInputPlane)
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
        bias.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
      case Xavier =>
        val fanIn = nInputPlane * kH * kW
        val fanOut = nOutputPlane * kH * kW
        val stdv = math.sqrt(6.0 / (fanIn + fanOut))
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
        bias.fill(ev.fromType(0))
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4, "Only support 3D or 4D(batch mode) input")
    require(input.isContiguous())

    if (weightMM == null) {
      weightMM = weight.view(nGroup, nOutputPlane / nGroup, nInputPlane * kH * kW / nGroup)
    }
    val dimWidth = if (input.dim() == 3) 3 else 4
    val dimHeight = if (input.dim() == 3) 2 else 3

    val inputWidth = input.size(dimWidth)
    val inputHeight = input.size(dimHeight)

    val outputWidth = (inputWidth + 2 * padW - kW) / dW + 1
    val outputHeight = (inputHeight + 2 * padH - kH) / dH + 1

    if (onesBias.dim() != 1 || onesBias.size(1) != outputHeight * outputWidth) {
      onesBias.resize(Array(outputHeight * outputWidth)).fill(ev.fromType(1.0))
    }

    require(outputWidth >= 1 && outputHeight >= 1, "output size is too small")
    if (input.dim() == 3) {
      require(input.size(1) == nInputPlane)
      require(input.isContiguous())
      val contiguousInput = input.contiguous()
      output.resize(Array(nOutputPlane, outputHeight, outputWidth))
      fInput.resize(Array(nGroup, kW * kH * nInputPlane / nGroup, outputHeight * outputWidth))
      var g = 0
      while(g < nGroup) {
        updateOutputFrame(
          contiguousInput.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
          output.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          weightMM.select(1, g + 1),
          bias.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          fInput.select(1, g + 1),
          kW, kH, dW, dH,
          padW, padH,
          nInputPlane / nGroup, inputWidth, inputHeight,
          nOutputPlane / nGroup, outputWidth, outputHeight)
        g += 1
      }
    } else {
      require(input.size(2) == nInputPlane)
      val batchSize = input.size(1)
      output.resize(Array(batchSize, nOutputPlane, outputHeight, outputWidth))

      fInput.resize(Array(Engine.coresNum, nGroup, kW * kH * nInputPlane / nGroup,
        outputHeight * outputWidth))

      if (results == null || results.length != Engine.coresNum) {
        results = new Array[Future[Unit]](Engine.coresNum)
      }


      var i, j = 0
      val minJobNum: Int = batchSize / Engine.coresNum
      val remainJobNum: Int = batchSize - minJobNum * Engine.coresNum

      while (j < Engine.coresNum) {
        val _j = j
        results(j) = Future {
          var _i = 1
          val distJobNum: Int = minJobNum + (if (_j < remainJobNum) 1 else 0)
          val indexS: Int = _j * minJobNum + (if (_j < remainJobNum) _j else remainJobNum)
          while (_i <= distJobNum) {
            val inputT = input.select(1, _i + indexS).contiguous()
            val outputT = output.select(1, _i + indexS)
            val fInputT = fInput.select(1, _j+1)
            var g = 0
            while (g < nGroup) {
              updateOutputFrame(
                inputT.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
                outputT.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
                weightMM.select(1, g + 1),
                bias.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
                fInputT.select(1, g + 1),
                kW, kH, dW, dH,
                padW, padH,
                nInputPlane / nGroup, inputWidth, inputHeight,
                nOutputPlane / nGroup, outputWidth, outputHeight)
              g += 1
            }
            _i += 1
          }
        }(Engine.getInstance())
        j += 1
      }


      i = 0
      while (i < results.length) {
        Await.result(results(i), Duration.Inf)
        i += 1
      }
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 3 || input.nDimension() == 4, "Only support 3D or 4D input")
    gradInput.resizeAs(input)

    val dimWidth = if (input.dim() == 3) 3 else 4
    val dimHeight = if (input.dim() == 3) 2 else 3

    val inputWidth = input.size(dimWidth)
    val inputHeight = input.size(dimHeight)

    val outputWidth = (inputWidth + 2 * padW - kW) / dW + 1
    val outputHeight = (inputHeight + 2 * padH - kH) / dH + 1


    //fGradInput.resizeAs(fInput)

    if (input.nDimension() == 3) {
      require(gradOutput.isContiguous())
      fGradInput.resize(Array(nGroup, kW * kH * nInputPlane / nGroup, outputHeight * outputWidth))
      val contiguousGradOutput = gradOutput.contiguous()
      var g = 0
      while(g < nGroup) {
        updateGradInputFrame(
          gradInput.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
          contiguousGradOutput.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          weightMM.select(1, g + 1).transpose(1, 2),
          fGradInput.select(1, g + 1),
          kW, kH, dW, dH, padW, padH)
        g += 1
      }
    } else {
      val batchSize = input.size(1)

      fGradInput.resize(Array(Engine.coresNum, nGroup, kW * kH * nInputPlane / nGroup,
        outputHeight * outputWidth))


      var i, j = 0
      val minJobNum: Int = batchSize / Engine.coresNum
      val remainJobNum: Int = batchSize - minJobNum * Engine.coresNum

      while (j < Engine.coresNum) {
        val _j = j
        results(j) = Future {
          var _i = 1
          val distJobNum: Int = minJobNum + (if (_j < remainJobNum) 1 else 0)
          val indexS: Int = _j * minJobNum + (if (_j < remainJobNum) _j else remainJobNum)
          while (_i <= distJobNum) {
            val gradInputT = gradInput.select(1, _i+indexS)
            val gradOutputT = gradOutput.select(1, _i+indexS).contiguous()
            val fgradInputT = fGradInput.select(1, _j+1)
            var g = 0
            while (g < nGroup) {
              updateGradInputFrame(
                gradInputT.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
                gradOutputT.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
                weightMM.select(1, g + 1).transpose(1, 2),
                fgradInputT.select(1, g + 1),
                kW, kH, dW, dH, padW, padH)
              g += 1
            }
            _i += 1
          }
        }(Engine.getInstance())
        j += 1
      }
      i = 0
      while (i < results.length) {
        Await.result(results(i), Duration.Inf)
        i += 1
      }
    }

    return gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
    scale: Double = 1.0): Unit = {
    require(input.nDimension() == 3 || input.nDimension() == 4, "Only support 3D or 4D input")
    val contiguousGradOutput = gradOutput.contiguous()


    val dimWidth = if (input.dim() == 3) 3 else 4
    val dimHeight = if (input.dim() == 3) 2 else 3

    val inputWidth = input.size(dimWidth)
    val inputHeight = input.size(dimHeight)

    val outputWidth = (inputWidth + 2 * padW - kW) / dW + 1
    val outputHeight = (inputHeight + 2 * padH - kH) / dH + 1


    if (input.nDimension() == 3) {
      if (gradWeightMM == null) {
        gradWeightMM = gradWeight.view(nGroup, nOutputPlane / nGroup,
          nInputPlane * kH * kW / nGroup)
      }
      fInput.resize(Array(nGroup, kW * kH * nInputPlane / nGroup, outputHeight * outputWidth))
      var g = 0
      while(g < nGroup) {
        accGradParametersFrame(
          contiguousGradOutput.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          gradWeightMM.select(1, g + 1),
          gradBias.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          fInput.select(1, g + 1),
          ev.fromType[Double](scale))
        g += 1
      }
    } else {
      val batchSize = input.size(1)

      fInput.resize(Array(Engine.coresNum, nGroup, kW * kH * nInputPlane / nGroup,
        outputHeight * outputWidth))

      if (gradientBiasMT == null) {
        gradWeightMM = Tensor[T]().resize(Array(batchSize, nGroup, nOutputPlane / nGroup,
          nInputPlane * kH * kW / nGroup))
        gradientBiasMT = Tensor[T]().resize(Array(batchSize, nOutputPlane))
      } else {
        gradWeightMM.resize(Array(batchSize, nGroup, nOutputPlane / nGroup,
          nInputPlane * kH * kW / nGroup))
      }

      if (ones.dim() != 1 || ones.size(1) != gradOutput.size(3) * gradOutput.size(4)) {
        ones.resize(Array(gradOutput.size(3) * gradOutput.size(4))).fill(ev.fromType(1.0))
      }

      if (onesBatch.dim() != 1 || onesBatch.size(1) != batchSize) {
        onesBatch.resize(Array(batchSize)).fill(ev.fromType(1.0))
      }


      var i, j = 0
      val minJobNum: Int = batchSize / Engine.coresNum
      val remainJobNum: Int = batchSize - minJobNum * Engine.coresNum

      while (j < Engine.coresNum) {
        val _j = j
        results(j) = Future {
          var _i = 1
          val distJobNum: Int = minJobNum + (if (_j < remainJobNum) 1 else 0)
          val indexS: Int = _j * minJobNum + (if (_j < remainJobNum) _j else remainJobNum)
          while (_i <= distJobNum) {
            val gradOutputT = contiguousGradOutput.select(1, _i+indexS)
            val inputT = input.select(1, _i+indexS).contiguous()
            val fInputT = fInput.select(1, _j+1)
            var g = 0
            while (g < nGroup) {
              write2fInput(
                inputT.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
                fInputT.select(1, g + 1),
                kW, kH, dW, dH,
                padW, padH,
                nInputPlane / nGroup, inputWidth, inputHeight,
                nOutputPlane / nGroup, outputWidth, outputHeight)
              calcGradParametersFrame(
                gradOutputT.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
                gradWeightMM.select(1, _i+indexS).select(1, g + 1),
                gradientBiasMT.select(1, _i+indexS).narrow(1, g * nOutputPlane / nGroup + 1,
                  nOutputPlane / nGroup),
                fInputT.select(1, g + 1),
                ev.fromType[Double](scale))
              g += 1
            }
            _i += 1
          }
        }(Engine.getInstance())
        j += 1
      }
      i = 0
      while (i < results.length) {
        Await.result(results(i), Duration.Inf)
        i += 1
      }

      val gradView = gradWeightMM.view(batchSize, nOutputPlane * nInputPlane * kH * kW / nGroup).t
      val grad = gradWeight.view(nOutputPlane * nInputPlane * kH * kW / nGroup)
      grad.addmv(ev.fromType(1.0), ev.fromType(1.0), gradView, onesBatch)
      gradBias.addmv(ev.fromType(1.0), ev.fromType(1.0), gradientBiasMT.t, onesBatch)
    }
  }

  override def updateParameters(learningRate: T): Unit = {
    weight.map(gradWeight, (a, b) => ev.minus(a, ev.times(learningRate, b)))
    bias.map(gradBias, (a, b) => ev.minus(a, ev.times(learningRate, b)))
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    gradBias.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialConvolution[T]]) {
      return false
    }
    val other = obj.asInstanceOf[SpatialConvolution[T]]
    if (this.eq(other)) {
      return true
    }

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

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + nInputPlane.hashCode()
    hash = hash * seed + nOutputPlane.hashCode()
    hash = hash * seed + kW.hashCode()
    hash = hash * seed + kH.hashCode()
    hash = hash * seed + dW.hashCode()
    hash = hash * seed + dH.hashCode()
    hash = hash * seed + padW.hashCode()
    hash = hash * seed + padH.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()

    hash
  }

  override def toString(): String = {
    s"nn.SpatialConvolution($nInputPlane -> $nOutputPlane, $kW x $kH, $dW, $dH, $padW, $padH)"
  }

  override def findModel(paramOffset: Int,
    indexes: Array[Int]): (Module[T], Int, Array[Int]) = {
    (this, paramOffset - nOutputPlane * nInputPlane * kH * kW - nOutputPlane, indexes)
  }


  private def write2fInput(input: Tensor[T], fInput: Tensor[T],
                                kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int,
                                nInputPlane: Int, inputWidth: Int, inputHeight: Int,
                                nOutputPlane: Int, outputWidth: Int, outputHeight: Int)(
                                 implicit ev: TensorNumeric[T]): Unit = {

    ev.getType() match {
      case "Double" =>
        val before = System.nanoTime()
        NNPrimitive.im2colDouble(fInput.asInstanceOf[Tensor[Double]],
          input.asInstanceOf[Tensor[Double]], kW, kH, dW, dH, padW, padH, nInputPlane,
          inputWidth, inputHeight, outputWidth, outputHeight)
        im2colTime += System.nanoTime() - before
      case "Float" =>
        val before = System.nanoTime()
        NNPrimitive.im2colFloat(fInput.asInstanceOf[Tensor[Float]],
          input.asInstanceOf[Tensor[Float]], kW, kH, dW, dH, padW, padH, nInputPlane,
          inputWidth, inputHeight, outputWidth, outputHeight)
        im2colTime += System.nanoTime() - before
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
  }

  private def updateOutputFrame(input: Tensor[T], output: Tensor[T], weight: Tensor[T],
    bias: Tensor[T], fInput: Tensor[T],
    kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int,
    nInputPlane: Int, inputWidth: Int, inputHeight: Int,
    nOutputPlane: Int, outputWidth: Int, outputHeight: Int)(
    implicit ev: TensorNumeric[T]): Unit = {

    val output2d = output.view(nOutputPlane, outputHeight * outputWidth)
    ev.getType() match {
      case "Double" =>
        val before = System.nanoTime()
        NNPrimitive.im2colDouble(fInput.asInstanceOf[Tensor[Double]],
          input.asInstanceOf[Tensor[Double]], kW, kH, dW, dH, padW, padH, nInputPlane,
          inputWidth, inputHeight, outputWidth, outputHeight)
        im2colTime += System.nanoTime() - before
      case "Float" =>
        val before = System.nanoTime()
        NNPrimitive.im2colFloat(fInput.asInstanceOf[Tensor[Float]],
          input.asInstanceOf[Tensor[Float]], kW, kH, dW, dH, padW, padH, nInputPlane,
          inputWidth, inputHeight, outputWidth, outputHeight)
        im2colTime += System.nanoTime() - before
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
    output2d.addmm(ev.fromType[Int](0), output2d, ev.fromType[Int](1), weight, fInput)
    output2d.addr(ev.fromType(1), bias, onesBias)
  }

  private def updateGradInputFrame(gradInput: Tensor[T], gradOutput: Tensor[T],
    weight: Tensor[T], fgradInput: Tensor[T], kW: Int, kH: Int, dW: Int, dH: Int,
    padW: Int, padH: Int)(implicit ev: TensorNumeric[T]): Unit = {

    ev.getType() match {
      case "Double" =>
        val gradOutput2d = Tensor(gradOutput.storage().asInstanceOf[Storage[Double]],
          gradOutput.storageOffset(), Array(gradOutput.size(1),
            gradOutput.size(2) * gradOutput.size(3)))
        fgradInput.asInstanceOf[Tensor[Double]].addmm(0.0, fgradInput.asInstanceOf[Tensor[Double]],
          1.0, weight.asInstanceOf[Tensor[Double]], gradOutput2d)
        gradInput.asInstanceOf[Tensor[Double]].zero()
        val before = System.nanoTime()
        NNPrimitive.col2imDouble(fgradInput.asInstanceOf[Tensor[Double]],
          gradInput.asInstanceOf[Tensor[Double]], kW, kH, dW, dH, padW, padH, gradInput.size(1),
          gradInput.size(3),
          gradInput.size(2), gradOutput.size(3), gradOutput.size(2))
        col2imTime += System.nanoTime() - before
      case "Float" =>
        val gradOutput2d = Tensor(gradOutput.storage().asInstanceOf[Storage[Float]],
          gradOutput.storageOffset(),
          Array(gradOutput.size(1), gradOutput.size(2) * gradOutput.size(3)))
        fgradInput.asInstanceOf[Tensor[Float]].addmm(0.0f, fgradInput.asInstanceOf[Tensor[Float]],
          1.0f, weight.asInstanceOf[Tensor[Float]], gradOutput2d)
        gradInput.asInstanceOf[Tensor[Float]].zero()
        val before = System.nanoTime()
        NNPrimitive.col2imFloat(fgradInput.asInstanceOf[Tensor[Float]],
          gradInput.asInstanceOf[Tensor[Float]], kW, kH, dW, dH, padW, padH, gradInput.size(1),
          gradInput.size(3),
          gradInput.size(2), gradOutput.size(3), gradOutput.size(2))
        col2imTime += System.nanoTime() - before
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
  }

  private def accGradParametersFrame(gradOutput: Tensor[T], gradWeight: Tensor[T],
    gradBias: Tensor[T], fInput: Tensor[T], scale: T)(implicit ev: TensorNumeric[T]): Unit = {

    ev.getType() match {
      case "Double" =>
        val gradOutput2d = Tensor[Double](gradOutput.storage().asInstanceOf[Storage[Double]],
          gradOutput.storageOffset(),
          Array(gradOutput.size(1), gradOutput.size(2) * gradOutput.size(3)))

        gradWeight.asInstanceOf[Tensor[Double]].addmm(1.0, gradWeight.asInstanceOf[Tensor[Double]],
          ev.toType[Double](scale), gradOutput2d,
          fInput.t.asInstanceOf[Tensor[Double]])

        var i = 0
        while (i < gradBias.size(1)) {
          var sum = 0.0
          val data = gradOutput2d.storage().array()
          val offset = gradOutput2d.storageOffset() - 1 + i * gradOutput2d.stride(1)
          var k = 0
          while (k < gradOutput2d.size(2)) {
            sum += data(k + offset)
            k += 1
          }
          gradBias.asInstanceOf[Tensor[Double]].setValue(
            i + 1, gradBias.asInstanceOf[Tensor[Double]].valueAt(i + 1) +
              (ev.toType[Double](scale) * sum))
          i += 1
        }
      case "Float" =>
        val gradOutput2d = Tensor[Float](gradOutput.storage().asInstanceOf[Storage[Float]],
          gradOutput.storageOffset(),
          Array(gradOutput.size(1), gradOutput.size(2) * gradOutput.size(3)))

        gradWeight.asInstanceOf[Tensor[Float]].addmm(1.0f, gradWeight.asInstanceOf[Tensor[Float]],
          ev.toType[Float](scale), gradOutput2d,
          fInput.t.asInstanceOf[Tensor[Float]])

        var i = 0
        while (i < gradBias.size(1)) {
          var sum = 0.0f
          val data = gradOutput2d.storage().array()
          val offset = gradOutput2d.storageOffset() - 1 + i * gradOutput2d.stride(1)
          var k = 0
          while (k < gradOutput2d.size(2)) {
            sum += data(k + offset)
            k += 1
          }
          gradBias.asInstanceOf[Tensor[Float]].setValue(
            i + 1, gradBias.asInstanceOf[Tensor[Float]].valueAt(i + 1) +
              (ev.toType[Float](scale) * sum))
          i += 1
        }
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
  }

  private def calcGradParametersFrame(gradOutput: Tensor[T], gradWeight: Tensor[T],
    gradBias: Tensor[T],
    fInput: Tensor[T], scale: T)(implicit ev: TensorNumeric[T]): Unit = {

    ev.getType() match {
      case "Double" =>
        val gradOutput2d = Tensor[Double](gradOutput.storage().asInstanceOf[Storage[Double]],
          gradOutput.storageOffset(),
          Array(gradOutput.size(1), gradOutput.size(2) * gradOutput.size(3)))

        gradWeight.asInstanceOf[Tensor[Double]].addmm(0.0, gradWeight.asInstanceOf[Tensor[Double]],
          ev.toType[Double](scale), gradOutput2d,
          fInput.t.asInstanceOf[Tensor[Double]])
        gradBias.asInstanceOf[Tensor[Double]].addmv(0.0, 1.0, gradOutput2d,
          ones.asInstanceOf[Tensor[Double]])
      case "Float" =>
        val gradOutput2d = Tensor[Float](gradOutput.storage().asInstanceOf[Storage[Float]],
          gradOutput.storageOffset(),
          Array(gradOutput.size(1), gradOutput.size(2) * gradOutput.size(3)))

        gradWeight.asInstanceOf[Tensor[Float]].addmm(0.0f, gradWeight.asInstanceOf[Tensor[Float]],
          ev.toType[Float](scale), gradOutput2d,
          fInput.t.asInstanceOf[Tensor[Float]])

        gradBias.asInstanceOf[Tensor[Float]].addmv(0.0f, 1.0f, gradOutput2d,
          ones.asInstanceOf[Tensor[Float]])
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
  }
}
