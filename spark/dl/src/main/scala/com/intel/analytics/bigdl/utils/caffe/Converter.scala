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
package com.intel.analytics.bigdl.utils.caffe

import caffe.Caffe
import caffe.Caffe._
import caffe.Caffe.EltwiseParameter.EltwiseOp
import caffe.Caffe.LRNParameter.NormRegion
import caffe.Caffe.PoolingParameter.PoolMethod
import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
/**
 * An abstract class to define interfaces when loading from/to caffe models
 * Caffe supports two kinds of layer definition LayerParameter & V1LayerParameter
 * Implementation [[V1LayerConverter]] and [[LayerConverter]]
 * V1LayerParameter is not recommended any more but we need to support old-versioned model
 */
abstract class Converter[T: ClassTag](implicit ev: TensorNumeric[T]) {

  // support user to customized BigDL compatible module to support those we have no mappings now
  private val customizedConverter =
    new mutable.HashMap[String, (GeneratedMessage) => Seq[ModuleNode[T]]]()

  // a caffe type to converter function mappings
  private val caffe2BigDL = new mutable.HashMap[String, (GeneratedMessage) => Seq[ModuleNode[T]]]()

  init

  def registerCutomizedConverter(layerType : String,
    converter : (GeneratedMessage) => Seq[ModuleNode[T]])
    : Unit = {
    require(!caffe2BigDL.contains(layerType), s"$layerType is already supported")
    require(!customizedConverter.contains(layerType), s"$layerType is already customized")
    customizedConverter(layerType) = converter
  }
  /**
   * Support customized layer mapping implemented by user for specific type
   */
  private def tryCustomizedConverter(layerType : String, layer : GeneratedMessage) :
    Seq[ModuleNode[T]] = {
    if (customizedConverter.contains(layerType)) {
      return customizedConverter(layerType)(layer)
    }
    throw new CaffeConversionException(s"$layerType is not supported in BigDL for now")
  }

  def convertLayerFromCaffe(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    val layerType = getLayerType(layer).toUpperCase
    if (caffe2BigDL.contains(layerType)) {
      if (caffe2BigDL(layerType) != null) caffe2BigDL(layerType)(layer)
      else null
    } else {
      tryCustomizedConverter(layerType, layer)
    }
  }

  protected def fromCaffeReLU(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(ReLU(true).setName(layerName).inputs())
  }

  private def fromCaffeLRN(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    val param = getLRNParam(layer).get
    val localSize = param.getLocalSize
    val alpha = param.getAlpha
    val belta = param.getBeta
    val k = param.getK
    val normRegion = param.getNormRegion
    normRegion match {
      case NormRegion.ACROSS_CHANNELS =>
        Seq(SpatialCrossMapLRN[T](localSize, alpha, belta, k).setName(layerName).inputs())
      case NormRegion.WITHIN_CHANNEL =>
        Seq(SpatialWithinChannelLRN[T](localSize, alpha, belta).setName(layerName).inputs())
      case _ => null
    }
  }

  private def fromCaffePooling(layer : GeneratedMessage): Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    val param = getPoolingParam(layer).get
    var kw = param.getKernelW
    var kh = param.getKernelH
    var dw = param.getStrideW
    var dh = param.getStrideH
    var pw = param.getPadW
    var ph = param.getPadH
    if (kw ==0 || kh == 0) {
      kw = param.getKernelSize
      kh = kw
    }
    if (dw == 0 || dh == 0) {
      dw = param.getStride
      dh = dw
    }
    if (pw == 0 || ph == 0) {
      pw = param.getPad
      ph = pw
    }
    val poolingType = param.getPool
    // caffe use ceil model
    val pooling = poolingType match {
      case PoolMethod.MAX => SpatialMaxPooling[T](kw, kh, dw, dh, pw, ph).ceil().
        setName(layerName).inputs()
      case PoolMethod.AVE => SpatialAveragePooling[T](kw, kh, dw, dh, pw, ph,
        param.getGlobalPooling).ceil().
        setName(layerName).inputs()
      case _ => null
    }
    Seq(pooling)
  }

  private def fromCaffeDropout(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = getDropoutParam(layer).get
    val layerName = getLayerName(layer)
    val initP = param.getDropoutRatio
    Seq(Dropout[T](initP).setName(layerName).inputs())
  }

  private def fromCaffeSoftmax(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(SoftMax().setName(layerName).inputs())
  }

  private def fromCaffeTanh(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(Tanh[T]().setName(layerName).inputs())
  }

  private def fromCaffeSigmoid(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(Sigmoid[T]().setName(layerName).inputs())
  }

  private def fromCaffeAbsVal(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(Abs[T]().setName(layerName).inputs())
  }

  private def fromCaffeConcat(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    val param = getConcatParam(layer)
    val dim = param.get.getAxis
    Seq(JoinTable[T](dim + 1, 0).setName(layerName).inputs())
  }

  private def fromCaffeFlatten(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(InferReshape[T](Array(0, -1)).setName(layerName).inputs())
  }

  private def fromCaffeLog(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(Log[T]().setName(layerName).inputs())
  }

  private def fromCaffePower(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    val param = getPowerParam(layer).get
    val power = param.getPower
    var scale = 1.0
    var shift = 0.0
    if (param.hasScale) scale = param.getScale
    if (param.hasShift) shift = param.getShift
    Seq(Power[T](power, scale, shift).setName(layerName).inputs())
  }

  private def fromCaffePreLU(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    val weightBlob = getBlob(layer, 0)
    sanityBlobCheck(layer, "weight", weightBlob)
    val weight = weightBlob.get
    val nOutPlane = if (weight.hasShape) weight.getShape.getDim(0)
    else weight.getNum
    Seq(PReLU[T](nOutPlane.toInt).setName(layerName).inputs())
  }

  private def fromCaffeRecurrent(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(Recurrent[T]().setName(layerName).inputs())
  }

  private def fromCaffeThreshold(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = getThresholdParam(layer).get
    var threshold = 1e-6
    if (param.hasThreshold) {
      threshold = param.getThreshold
    }
    Seq(BinaryThreshold[T](threshold).setName(getLayerName(layer)).inputs())
  }

  private def fromCaffeExp(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(Exp[T]().setName(layerName).inputs())
  }

  private def fromCaffeSlice(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = getSliceParam(layer)
    val layerName = getLayerName(layer)
    val axis = param.get.getAxis
    Seq(SplitTable[T](axis).setName(layerName).inputs())
  }

  private def fromCaffeEltwise(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = getEltWiseParam(layer).get
    val layerName = getLayerName(layer)
    val opsType = param.getOperation
    val ops = opsType match {
      case EltwiseOp.PROD => CMulTable[T]().setName(layerName).inputs()
      case EltwiseOp.MAX => CMaxTable[T]().setName(layerName).inputs()
      case EltwiseOp.SUM =>
        val coeff1 = if (param.getCoeffCount == 0) 1 else param.getCoeff(0)
        val coeff2 = if (param.getCoeffCount == 0) 1 else param.getCoeff(1)
        if (coeff1 == 1 && coeff2 == 1) {
          CAddTable[T]().setName(layerName).inputs()
        } else if (coeff1 == 1 && coeff2 == -1) {
          CSubTable[T]().setName(layerName).inputs()
        } else {
          val mul1 = MulConstant[T](coeff1.toFloat).inputs()
          val mul2 = MulConstant[T](coeff2.toFloat).inputs()
          val caddTable = CAddTable[T]().setName(layerName).inputs(mul1, mul2)
          Graph[T](Array(mul1, mul2), Array(caddTable)).inputs()
        }
      case _ => null
    }
    Seq(ops)
  }

  protected def fromCaffeBatchNormalization(layer : GeneratedMessage) : Seq[ModuleNode[T]]

  protected def fromCaffeConvolution(layer : GeneratedMessage) : Seq[ModuleNode[T]]

  protected def fromCaffeInnerProduct(layer : GeneratedMessage) : Seq[ModuleNode[T]]

  protected def fromCaffeELU(layer : GeneratedMessage) : Seq[ModuleNode[T]]

  protected def fromCaffeReshape(layer : GeneratedMessage) : Seq[ModuleNode[T]]

  protected def fromCaffeScale(layer : GeneratedMessage) : Seq[ModuleNode[T]]

  protected def fromCaffeBias(layer : GeneratedMessage) : Seq[ModuleNode[T]]

  protected def fromCaffeTile(layer : GeneratedMessage) : Seq[ModuleNode[T]]

  protected def fromCaffeInput(layer: GeneratedMessage): Seq[ModuleNode[T]]

  protected def getLayerType(layer : GeneratedMessage) : String

  protected def getLayerName(layer : GeneratedMessage) : String

  protected def getConvolutionParam(layer : GeneratedMessage): Option[ConvolutionParameter]

  protected def getLRNParam(layer : GeneratedMessage): Option[LRNParameter]

  protected def getPoolingParam(layer : GeneratedMessage): Option[PoolingParameter]

  protected def getInnerProductParam(layer : GeneratedMessage): Option[InnerProductParameter]

  protected def getDropoutParam(layer : GeneratedMessage): Option[DropoutParameter]

  protected def getConcatParam(layer : GeneratedMessage) : Option[ConcatParameter]

  protected def getPowerParam(layer : GeneratedMessage) : Option[PowerParameter]

  protected def getThresholdParam(layer : GeneratedMessage): Option[ThresholdParameter]

  protected def getSliceParam(layer : GeneratedMessage): Option[SliceParameter]

  protected def getEltWiseParam(layer : GeneratedMessage): Option[EltwiseParameter]

  def toCaffe(moduleNode : AbstractModule[Activity, Activity, T],
              bottoms : ArrayBuffer[String], nextSize : Int) : Seq[GeneratedMessage] = {
    val module = moduleNode.asInstanceOf[AbstractModule[_, _, _]]
    val model : Seq[GeneratedMessage] = module match {
      case convolution : SpatialConvolution[_] => toCaffeConvolution(moduleNode, bottoms, nextSize)
      case deconv : SpatialFullConvolution[_] => toCaffeDeConvolution(moduleNode, bottoms, nextSize)
      case relu : ReLU[_] => toCaffeRelu(moduleNode, bottoms, nextSize)
      case crossMapLrn : SpatialCrossMapLRN[_] => toCaffeLRN(moduleNode, bottoms, nextSize)
      case inChannelLrn : SpatialWithinChannelLRN[_] => toCaffeLRN(moduleNode, bottoms, nextSize)
      case maxPooling : SpatialMaxPooling[_] => toCaffeMaxPooling(moduleNode, bottoms, nextSize)
      case avgPooling : SpatialAveragePooling[_] => toCaffeAvePooling(moduleNode, bottoms, nextSize)
      case linear : Linear[_] => toCaffeInnerProduct(moduleNode, bottoms, nextSize)
      case dropout : Dropout[_] => toCaffeDropOut(moduleNode, bottoms, nextSize)
      case logSoftMax : LogSoftMax[_] => toCaffeLogSoftMax(moduleNode, bottoms, nextSize)
      case tanh : Tanh[_] => toCaffeTanh(moduleNode, bottoms, nextSize)
      case sigmoid : Sigmoid[_] => toCaffeSigmoid(moduleNode, bottoms, nextSize)
      case abs : Abs[_] => toCaffeAbs(moduleNode, bottoms, nextSize)
      case bartchNorm : SpatialBatchNormalization[_] =>
        toCaffeBatchNormalization(moduleNode, bottoms, nextSize)
      case joinTable : JoinTable[_] => toCaffeConcat(moduleNode, bottoms, nextSize)
      case elu : ELU[_] => toCaffeElu(moduleNode, bottoms, nextSize)
      case infershape : InferReshape[_] => toCaffeFlattern(moduleNode, bottoms, nextSize)
      case log : Log[_] => toCaffeLog(moduleNode, bottoms, nextSize)
      case power : Power[_] => toCaffePower(moduleNode, bottoms, nextSize)
      case prelu : PReLU[_] => toCaffePReLu(moduleNode, bottoms, nextSize)
      case recurrent : Recurrent[_] => toCaffeRecurrent(moduleNode, bottoms, nextSize)
      case reshape : Reshape[_] => toCaffeReshape(moduleNode, bottoms, nextSize)
      case scale : Scale[_] => toCaffeScale(moduleNode, bottoms, nextSize)
      case add : Add[_] => toCaffeBias(moduleNode, bottoms, nextSize)
      case threshold : Threshold[_] => toCaffeThreshold(moduleNode, bottoms, nextSize)
      case exp : Exp[_] => toCaffeExp(moduleNode, bottoms, nextSize)
      case splitTable : SplitTable[_] => toCaffeSlice(moduleNode, bottoms, nextSize)
      case replicate : Replicate[_] => toCaffeTile(moduleNode, bottoms, nextSize)
      case cmax : CMaxTable[_] => toCaffeEltWiseMax(moduleNode, bottoms, nextSize)
      case cadd : CAdd[_] => toCaffeEltWiseAdd(moduleNode, bottoms, nextSize)
      case csub : CSubTable[_] => toCaffeEltWiseSub(moduleNode, bottoms, nextSize)
      case sequantial : Sequential[_] => toCaffeSequential(moduleNode, bottoms, nextSize)
      case _ => throw  new CaffeConversionException(s"${moduleNode} is not supported")
    }
    model
  }

  protected def toCaffeConvolution(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeDeConvolution(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeRelu(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeLRN(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeMaxPooling(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeAvePooling(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeInnerProduct(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeDropOut(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeLogSoftMax(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeTanh(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeSigmoid(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeAbs(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeBatchNormalization(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeConcat(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeElu(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeFlattern(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeLog(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffePower(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffePReLu(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeRecurrent(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeReshape(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeScale(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeBias(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeThreshold(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeExp(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeSlice(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeTile(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeEltWiseMax(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeEltWiseAdd(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeEltWiseSub(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeSequential(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage]

  protected def toCaffeConvolutionParam(module : AbstractModule[Activity, Activity, T])
  : mutable.HashMap[String, Int] = {
    val map = new mutable.HashMap[String, Int]()
    val layer = classOf[SpatialConvolution[T]].cast(module)
    val nInputPlane = layer.nInputPlane
    val nOutputPlane = layer.nOutputPlane
    val kernelW = layer.kernelW
    val kernelH = layer.kernelH
    val strideW = layer.strideW
    val strideH = layer.strideH
    val padW = layer.padW
    val padH = layer.padH
    val ngroup = layer.nGroup
    map("nInputPlane") = nInputPlane
    map("nOutputPlane") = nOutputPlane
    map("kernelW") = kernelW
    map("kernelH") = kernelH
    map("strideW") = strideW
    map("strideH") = strideH
    map("padW") = padW
    map("padH") = padH
    map("ngroup") = ngroup
    map("withBias") = if (layer.withBias) 1 else 0
    map
  }

  protected def toCaffeDeConvolutionParam(module : AbstractModule[Activity, Activity, T])
  : mutable.HashMap[String, Int] = {
    val map = new mutable.HashMap[String, Int]()
    val layer = classOf[SpatialFullConvolution[T]].cast(module)
    if (layer.adjW != 0 || layer.adjH != 0) {
      throw new CaffeConversionException("Caffe doesn't support extra width/height amending")
    }
    val nInputPlane = layer.nOutputPlane
    val nOutputPlane = layer.nInputPlane
    val kernelW = layer.kW
    val kernelH = layer.kH
    val strideW = layer.dW
    val strideH = layer.dH
    val padW = layer.padW
    val padH = layer.padH
    val ngroup = layer.nGroup
    map("nInputPlane") = nInputPlane
    map("nOutputPlane") = nOutputPlane
    map("kernelW") = kernelW
    map("kernelH") = kernelH
    map("strideW") = strideW
    map("strideH") = strideH
    map("padW") = padW
    map("padH") = padH
    map("ngroup") = ngroup
    map("withBias") = if (layer.noBias) 0 else 1
    map
  }

  protected def toCaffeLRNParam(module : AbstractModule[Activity, Activity, T])
  : (Int, Double, Double, Double, String) = {
    if (module.isInstanceOf[SpatialCrossMapLRN[T]]) {
      val layer = classOf[SpatialCrossMapLRN[T]].cast(module)
      (layer.size, layer.alpha, layer.beta, layer.k, layer.getClass.getSimpleName)
    } else if (module.isInstanceOf[SpatialWithinChannelLRN[T]]) {
      val layer = classOf[SpatialWithinChannelLRN[T]].cast(module)
      (layer.size, layer.alpha, layer.beta, 0, layer.getClass.getSimpleName)
    } else {
      null
    }
  }

  protected def toCaffeMaxPoolingParam(module : AbstractModule[Activity, Activity, T])
  : PoolingParameter = {
    val layer = classOf[SpatialMaxPooling[T]].cast(module)
    val poolingParameter = PoolingParameter.newBuilder()
    poolingParameter.setKernelW(layer.kW)
    poolingParameter.setKernelH(layer.kH)
    poolingParameter.setStrideW(layer.dW)
    poolingParameter.setStrideH(layer.dH)
    poolingParameter.setPadW(layer.padW)
    poolingParameter.setPadH(layer.padH)
    poolingParameter.setPool(PoolMethod.MAX)
    poolingParameter.build
  }

  protected def toCaffeAvgPoolingParam(module : AbstractModule[Activity, Activity, T])
  : PoolingParameter = {
    val layer = classOf[SpatialAveragePooling[T]].cast(module)
    val poolingParameter = PoolingParameter.newBuilder()
    poolingParameter.setKernelW(layer.kW)
    poolingParameter.setKernelH(layer.kH)
    poolingParameter.setStrideW(layer.dW)
    poolingParameter.setStrideH(layer.dH)
    poolingParameter.setPadW(layer.padW)
    poolingParameter.setPadH(layer.padH)
    poolingParameter.setPool(PoolMethod.AVE)
    poolingParameter.setGlobalPooling(layer.globalPooling)
    poolingParameter.build
  }

  protected def toCaffeInnerProductParam(module : AbstractModule[Activity, Activity, T])
  : (Int, Int, Boolean) = {
    val layer = classOf[Linear[T]].cast(module)
    (layer.inputSize, layer.outputSize, layer.withBias)
  }

  protected def toCaffeDropOutParam(module : AbstractModule[Activity, Activity, T]) : Double = {
    val layer = classOf[Dropout[T]].cast(module)
    layer.initP
  }

  protected def toCaffeBatchNormParam(module : AbstractModule[Activity, Activity, T]) : Double = {
    val layer = classOf[BatchNormalization[T]].cast(module)
    layer.eps
  }

  protected def toCaffeConcatParam(module : AbstractModule[Activity, Activity, T]) : Int = {
    val layer = classOf[JoinTable[T]].cast(module)
    layer.dimension
  }

  protected def toCaffeEluParam(module : AbstractModule[Activity, Activity, T]) : ELUParameter = {
    val eLUParameter = ELUParameter.newBuilder()
    val layer = classOf[ELU[T]].cast(module)
    eLUParameter.setAlpha(layer.alpha.toFloat)
    eLUParameter.build()
  }

  protected def toCaffePowerParam(module : AbstractModule[Activity, Activity, T])
  : PowerParameter = {
    val powerParameter = PowerParameter.newBuilder
    val layer = classOf[Power[T]].cast(module)
    powerParameter.setPower(layer.power.toFloat)
    powerParameter.setScale(layer.scale.toFloat)
    powerParameter.setShift(layer.shift.toFloat)
    powerParameter.build
  }

  protected def toCaffeReshapeParam(module : AbstractModule[Activity, Activity, T])
  : ReshapeParameter = {
    val reshapeParameter = ReshapeParameter.newBuilder()
    val layer = classOf[Reshape[T]].cast(module)
    val size = layer.batchSize
    val shapeBlob = BlobShape.newBuilder
    var i = 0
    while (i < size.length) {
      shapeBlob.setDim(0, size(i))
      i += 1
    }
    reshapeParameter.setShape(shapeBlob.build)
    reshapeParameter.build
  }

  protected def toCaffeScalaParam(module : AbstractModule[Activity, Activity, T]) : BlobShape = {
    val layer = classOf[Scale[T]].cast(module)
    val size = layer.size
    val shapeBlob = BlobShape.newBuilder
    var i = 0
    while (i < size.length) {
      shapeBlob.setDim(i, size(i))
      i += 1
    }
    shapeBlob.build
  }

  protected def toCaffeThresholdParam(module : AbstractModule[Activity, Activity, T])
  : ThresholdParameter = {
    val layer = classOf[Threshold[T]].cast(module)
    val threshold = layer.threshold
    val thresholdParameter = ThresholdParameter.newBuilder
    thresholdParameter.setThreshold(threshold.toFloat)
    thresholdParameter.build
  }

  protected def toCaffeSliceParam(module : AbstractModule[Activity, Activity, T])
  : SliceParameter = {
    val layer = classOf[SplitTable[T]].cast(module)
    val axis = layer.dimension
    val sliceParameter = SliceParameter.newBuilder
    sliceParameter.setAxis(axis)
    sliceParameter.build
  }

  protected def toCaffeTileParam(module : AbstractModule[Activity, Activity, T])
  : TileParameter = {
    val layer = classOf[Replicate[T]].cast(module)
    val tile = layer.nFeatures
    val axis = layer.dim
    val tileParameter = TileParameter.newBuilder
    tileParameter.setTiles(tile)
    tileParameter.setAxis(axis)
    tileParameter.build
  }

  protected def getBlob(layer : GeneratedMessage, ind: Int): Option[Caffe.BlobProto]

  protected def sanityBlobCheck(layer : GeneratedMessage, blobInfo: String,
                                blob : Option[Caffe.BlobProto]) : Unit = {
    val name = getLayerName(layer)
    val tpe = getLayerType(layer)
    if (!blob.isDefined) {
      throw new CaffeConversionException(s"$tpe : $name missing $blobInfo in binary file")
    }
  }

  private def init() = {
    caffe2BigDL("CONVOLUTION") = fromCaffeConvolution
    caffe2BigDL("DECONVOLUTION") = fromCaffeConvolution
    caffe2BigDL("INNERPRODUCT") = fromCaffeInnerProduct
    caffe2BigDL("INNER_PRODUCT") = fromCaffeInnerProduct
    caffe2BigDL("RELU") = fromCaffeReLU
    caffe2BigDL("LRN") = fromCaffeLRN
    caffe2BigDL("POOLING") = fromCaffePooling
    caffe2BigDL("DROPOUT") = fromCaffeDropout
    caffe2BigDL("SOFTMAX") = fromCaffeSoftmax
    caffe2BigDL("SOFTMAX_LOSS") = null
    caffe2BigDL("SOFTMAXWITHLOSS") = null
    caffe2BigDL("TANH") = fromCaffeTanh
    caffe2BigDL("SIGMOID") = fromCaffeSigmoid
    caffe2BigDL("SIGMOIDCROSSENTROPYLOSS") = fromCaffeSigmoid
    caffe2BigDL("ABSVAL") = fromCaffeAbsVal
    caffe2BigDL("BATCHNORM") = fromCaffeBatchNormalization
    caffe2BigDL("CONCAT") = fromCaffeConcat
    caffe2BigDL("ELU") = fromCaffeELU
    caffe2BigDL("FLATTEN") = fromCaffeFlatten
    caffe2BigDL("LOG") = fromCaffeLog
    caffe2BigDL("POWER") = fromCaffePower
    caffe2BigDL("PRELU") = fromCaffePreLU
    caffe2BigDL("RECURRENT") = fromCaffeRecurrent
    caffe2BigDL("RNN") = fromCaffeRecurrent
    caffe2BigDL("RESHAPE") = fromCaffeReshape
    caffe2BigDL("SCALE") = fromCaffeScale
    caffe2BigDL("BIAS") = fromCaffeBias
    caffe2BigDL("THRESHOLD") = fromCaffeThreshold
    caffe2BigDL("EXP") = fromCaffeExp
    caffe2BigDL("SLICE") = fromCaffeSlice
    caffe2BigDL("TILE") = fromCaffeTile
    caffe2BigDL("ELTWISE") = fromCaffeEltwise
    caffe2BigDL("INPUT") = fromCaffeInput
    caffe2BigDL("DATA") = fromCaffeInput
    caffe2BigDL("DUMMYDATA") = fromCaffeInput
    caffe2BigDL("ANNOTATEDDATA") = fromCaffeInput
    caffe2BigDL("MEMORYDATA") = fromCaffeInput
    caffe2BigDL("ACCURACY") = null
    caffe2BigDL("SILENCE") = null
  }
}
