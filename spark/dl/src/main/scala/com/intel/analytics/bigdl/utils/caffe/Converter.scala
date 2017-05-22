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

import caffe.Caffe.EltwiseParameter.EltwiseOp
import caffe.Caffe.{BlobProto, _}
import caffe.Caffe.PoolingParameter.PoolMethod
import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

abstract class Converter[T: ClassTag](implicit ev: TensorNumeric[T]) {

  private var customizedLayer : Map[String, AbstractModule[Activity, Activity, T]] = _

  def setCustomizedLayer(map : Map[String, AbstractModule[Activity, Activity, T]]) : this.type = {
    customizedLayer = map
    this
  }
  /**
    * Support customized layer mapping implemented by user for specific type
   */
  private def fitCustomizedLayer(layerType : String, layerName : String) : ModuleNode[T] = {
    if (customizedLayer !=null && customizedLayer.contains(layerType)) {
      return customizedLayer(layerType).setName(layerName).apply()
    }
    throw new Exception(s"$layerType is not supported in BigDL fow now")
  }

  def convertLayerFromCaffe(layer : GeneratedMessage) : ModuleNode[T] = {
    val layerName = getLayerName(layer)
    val layerType = getLayerType(layer).toUpperCase
    val module = layerType match {
      case "CONVOLUTION" => fromCaffeConvolution(layer)
      case "RELU" => fromCaffeReLU(layer)
      case "LRN" => fromCaffeLRN(layer)
      case "POOLING" => fromCaffePooling(layer)
      case "INNERPRODUCT" => fromCaffeInnerProduct(layer)
      case "INNER_PRODUCT" => fromCaffeInnerProduct(layer)
      case "DROPOUT" => fromCaffeDropout(layer)
      case "SOFTMAX_LOSS" => fromCaffeSoftmax(layer)
      case "SOFTMAX" => fromCaffeSoftmax(layer)
      case "TANH" => fromCaffeTanh(layer)
      case "SIGMOID" => fromCaffeSigmoid(layer)
      case "SIGMOIDCROSSENTROPYLOSS" => fromCaffeSigmoid(layer)
      case "ABSVAL" => fromCaffeAbsVal(layer)
      case "BATCHNORM" => fromCaffeBatchNormalization(layer)
      case "CONCAT" => fromCaffeConcat(layer)
      case "ELU" => fromCaffeELU(layer)
      case "FLATTEN" => fromCaffeFlatten(layer)
      case "LOG" => fromCaffeLog(layer)
      case "POWER" => fromCaffePower(layer)
      case "PRELU" => fromCaffePreLU(layer)
      case "RECURRENT" => fromCaffeRecurrent(layer)
      case "RNN" => fromCaffeRecurrent(layer)
      case "RESHAPE" => fromCaffeReshape(layer)
      case "SCALE" => fromCaffeScale(layer)
      case "BIAS" => fromCaffeBias(layer)
      case "THRESHOLD" => fromCaffeThreshold(layer)
      case "Exp" => fromCaffeExp(layer)
      case "SLICE" => fromCaffeSlice(layer)
      case "TILE" => fromCaffeTile(layer)
      case "ELTWISE" => fromCaffeEltwise(layer)
      case "INPUT" => null
      case "DATA" => null
      case _ => fitCustomizedLayer(layerType, layerName)
    }
    module
  }

  protected def fromCaffeReLU(layer : GeneratedMessage) : ModuleNode[T] = {
    val layerName = getLayerName(layer)
    new ReLU(true).setName(layerName).apply()
  }

  private def fromCaffeLRN(layer : GeneratedMessage) : ModuleNode[T] = {
    val layerName = getLayerName(layer)
    val param = getLRNParam(layer).get
    val localSize = param.getLocalSize
    val alpha = param.getAlpha
    val belta = param.getBeta
    val k = param.getK
    new SpatialCrossMapLRN[T](localSize, alpha, belta, k).setName(layerName).apply()
  }

  private def fromCaffePooling(layer : GeneratedMessage): ModuleNode[T] = {
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
        setName(layerName).apply()
      case PoolMethod.AVE => SpatialAveragePooling[T](kw, kh, dw, dh, pw, ph).ceil().
        setName(layerName).apply()
      case _ => null
    }
    pooling
  }

  private def fromCaffeDropout(layer : GeneratedMessage) : ModuleNode[T] = {
    val param = getDropoutParam(layer).get
    val layerName = getLayerName(layer)
    val initP = param.getDropoutRatio
    new Dropout[T](initP).setName(layerName).apply()
  }

  private def fromCaffeSoftmax(layer : GeneratedMessage) : ModuleNode[T] = {
    val layerName = getLayerName(layer)
    new LogSoftMax().setName(layerName).apply()
  }

  private def fromCaffeTanh(layer : GeneratedMessage) : ModuleNode[T] = {
    val layerName = getLayerName(layer)
    new Tanh[T]().setName(layerName).apply()
  }

  private def fromCaffeSigmoid(layer : GeneratedMessage) : ModuleNode[T] = {
    val layerName = getLayerName(layer)
    new Sigmoid[T]().setName(layerName).apply()
  }

  private def fromCaffeAbsVal(layer : GeneratedMessage) : ModuleNode[T] = {
    val layerName = getLayerName(layer)
    new Abs[T]().setName(layerName).apply()
  }

  private def fromCaffeConcat(layer : GeneratedMessage) : ModuleNode[T] = {
    val layerName = getLayerName(layer)
    val param = getConcatParam(layer)
    val dim = param.get.getAxis
    JoinTable[T](dim + 1, 0).setName(layerName).apply()
  }

  private def fromCaffeFlatten(layer : GeneratedMessage) : ModuleNode[T] = {
    val layerName = getLayerName(layer)
    FlattenTable[T].setName(layerName).apply()
  }

  private def fromCaffeLog(layer : GeneratedMessage) : ModuleNode[T] = {
    val layerName = getLayerName(layer)
    Log[T]().setName(layerName).apply()
  }

  private def fromCaffePower(layer : GeneratedMessage) : ModuleNode[T] = {
    val layerName = getLayerName(layer)
    val param = getPowerParam(layer).get
    val power = param.getPower
    var scale = 1.0
    var shift = 0.0
    if (param.hasScale) scale = param.getScale
    if (param.hasShift) shift = param.getShift
    Power[T](power, scale, shift).apply()
  }

  private def fromCaffePreLU(layer : GeneratedMessage) : ModuleNode[T] = {
    val layerName = getLayerName(layer)
    PReLU[T]().setName(layerName).apply()
  }

  private def fromCaffeRecurrent(layer : GeneratedMessage) : ModuleNode[T] = {
    val layerName = getLayerName(layer)
    Recurrent[T]().setName(layerName).apply()
  }

  private def fromCaffeThreshold(layer : GeneratedMessage) : ModuleNode[T] = {
    val param = getThresholdParam(layer).get
    var threshold = 1e-6
    if (param.hasThreshold) {
      threshold = param.getThreshold
    }
    Threshold[T](threshold).setName(getLayerName(layer)).apply()
  }

  private def fromCaffeExp(layer : GeneratedMessage) : ModuleNode[T] = {
    val layerName = getLayerName(layer)
    Exp[T]().setName(layerName).apply()
  }

  private def fromCaffeSlice(layer : GeneratedMessage) : ModuleNode[T] = {
    val param = getSliceParam(layer)
    val layerName = getLayerName(layer)
    val axis = param.get.getAxis
    SplitTable[T](axis).setName(layerName).apply()
  }

  private def fromCaffeEltwise(layer : GeneratedMessage) : ModuleNode[T] = {
    val param = getEltWiseParam(layer).get
    val layerName = getLayerName(layer)
    val opsType = param.getOperation
    val coeff2 = param.getCoeff(1)
    val ops = opsType match {
      case EltwiseOp.PROD => CMaxTable[T]().setName(layerName).apply()
      case EltwiseOp.MAX => CMaxTable[T]().setName(layerName).apply()
      case EltwiseOp.SUM =>
        if (coeff2 < 0) {
          CAddTable[T]().setName(layerName).apply()
        } else {
          CSubTable[T]().setName(layerName).apply()
        }
      case _ => null
    }
    ops
  }

  protected def fromCaffeBatchNormalization(layer : GeneratedMessage) : ModuleNode[T]

  protected def fromCaffeConvolution(layer : GeneratedMessage) : ModuleNode[T]

  protected def fromCaffeInnerProduct(layer : GeneratedMessage) : ModuleNode[T]

  protected def fromCaffeELU(layer : GeneratedMessage) : ModuleNode[T]

  protected def fromCaffeReshape(layer : GeneratedMessage) : ModuleNode[T]

  protected def fromCaffeScale(layer : GeneratedMessage) : ModuleNode[T]

  protected def fromCaffeBias(layer : GeneratedMessage) : ModuleNode[T]

  protected def fromCaffeTile(layer : GeneratedMessage) : ModuleNode[T]

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

  def toCaffe(moduleNode : ModuleNode[T],
              bottoms : ArrayBuffer[String]) : GeneratedMessage = {
    val module = moduleNode.element.asInstanceOf[AbstractModule[_, _, _]]
    val model = module match {
      case convolution : SpatialConvolution[_] => toCaffeConvolution(moduleNode, bottoms)
      case  relu : ReLU[_] => toCaffeRelu(moduleNode, bottoms)
      case lrn : SpatialCrossMapLRN[_] => toCaffeLRN(moduleNode, bottoms)
      case maxPooling : SpatialMaxPooling[_] => toCaffeMaxPooling(moduleNode, bottoms)
      case avgPooling : SpatialAveragePooling[_] => toCaffeAvePooling(moduleNode, bottoms)
      case linear : Linear[_] => toCaffeInnerProduct(moduleNode, bottoms)
      case dropout : Dropout[_] => toCaffeDropOut(moduleNode, bottoms)
      case logSoftMax : LogSoftMax[_] => toCaffeLogSoftMax(moduleNode, bottoms)
      case tanh : Tanh[_] => toCaffeTanh(moduleNode, bottoms)
      case sigmoid : Sigmoid[_] => toCaffeSigmoid(moduleNode, bottoms)
      case abs : Abs[_] => toCaffeAbs(moduleNode, bottoms)
      case bartchNorm : BatchNormalization[_] => toCaffeBatchNormalization(moduleNode, bottoms)
      case joinTable : JoinTable[_] => toCaffeConcat(moduleNode, bottoms)
      case elu : ELU[_] => toCaffeElu(moduleNode, bottoms)
      case flatternTable : FlattenTable[_] => toCaffeFlattern(moduleNode, bottoms)
      case log : Log[_] => toCaffeLog(moduleNode, bottoms)
      case power : Power[_] => toCaffePower(moduleNode, bottoms)
      case prelu : PReLU[_] => toCaffePReLu(moduleNode, bottoms)
      case recurrent : Recurrent[_] => toCaffeRecurrent(moduleNode, bottoms)
      case reshape : Reshape[_] => toCaffeReshape(moduleNode, bottoms)
      case scale : Scale[_] => toCaffeScale(moduleNode, bottoms)
      case add : Add[_] => toCaffeBias(moduleNode, bottoms)
      case threshold : Threshold[_] => toCaffeThreshold(moduleNode, bottoms)
      case exp : Exp[_] => toCaffeExp(moduleNode, bottoms)
      case splitTable : SplitTable[_] => toCaffeSlice(moduleNode, bottoms)
      case replicate : Replicate[_] => toCaffeTile(moduleNode, bottoms)
      case cmax : CMaxTable[_] => toCaffeEltWiseMax(moduleNode, bottoms)
      case cadd : CAdd[_] => toCaffeEltWiseAdd(moduleNode, bottoms)
      case csub : CSubTable[_] => toCaffeEltWiseSub(moduleNode, bottoms)
      case _ => throw  new UnsupportedOperationException(s"${moduleNode.element} is not supported")
    }
    model
  }

  protected def toCaffeConvolution(moduleNode : ModuleNode[T],
                                   bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeRelu(moduleNode : ModuleNode[T],
                            bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeLRN(moduleNode : ModuleNode[T],
                            bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeMaxPooling(moduleNode : ModuleNode[T],
                           bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeAvePooling(moduleNode : ModuleNode[T],
                                  bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeInnerProduct(moduleNode : ModuleNode[T],
                                  bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeDropOut(moduleNode : ModuleNode[T],
                                    bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeLogSoftMax(moduleNode : ModuleNode[T],
                               bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeTanh(moduleNode : ModuleNode[T],
                                  bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeSigmoid(moduleNode : ModuleNode[T],
                            bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeAbs(moduleNode : ModuleNode[T],
                               bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeBatchNormalization(moduleNode : ModuleNode[T],
                           bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeConcat(moduleNode : ModuleNode[T],
                                          bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeElu(moduleNode : ModuleNode[T],
                              bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeFlattern(moduleNode : ModuleNode[T],
                           bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeLog(moduleNode : ModuleNode[T],
                                bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffePower(moduleNode : ModuleNode[T],
                           bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffePReLu(moduleNode : ModuleNode[T],
                             bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeRecurrent(moduleNode : ModuleNode[T],
                             bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeReshape(moduleNode : ModuleNode[T],
                                 bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeScale(moduleNode : ModuleNode[T],
                               bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeBias(moduleNode : ModuleNode[T],
                             bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeThreshold(moduleNode : ModuleNode[T],
                            bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeExp(moduleNode : ModuleNode[T],
                                 bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeSlice(moduleNode : ModuleNode[T],
                           bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeTile(moduleNode : ModuleNode[T],
                             bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeEltWiseMax(moduleNode : ModuleNode[T],
                            bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeEltWiseAdd(moduleNode : ModuleNode[T],
                                  bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeEltWiseSub(moduleNode : ModuleNode[T],
                                  bottoms : ArrayBuffer[String]): GeneratedMessage

  protected def toCaffeConvolutionParam(module : AbstractModule[Activity, Tensor[T], T])
    : mutable.HashMap[String, Int] = {
    var map = new mutable.HashMap[String, Int]()
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
    map
  }

  protected def toCaffeLRNParam(module : AbstractModule[Activity, Tensor[T], T])
    : (Int, Double, Double, Double) = {
    val layer = classOf[SpatialCrossMapLRN[T]].cast(module)
    (layer.size, layer.alpha, layer.beta, layer.k)
  }

  protected def toCaffeMaxPoolingParam(module : AbstractModule[Activity, Tensor[T], T])
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

  protected def toCaffeAvgPoolingParam(module : AbstractModule[Activity, Tensor[T], T])
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
    poolingParameter.build
  }

  protected def toCaffeInnerProductParam(module : AbstractModule[Activity, Tensor[T], T])
    : (Int, Int, Boolean) = {
    val layer = classOf[Linear[T]].cast(module)
    (layer.inputSize, layer.outputSize, layer.withBias)
  }

  protected def toCaffeDropOutParam(module : AbstractModule[Activity, Tensor[T], T]) : Double = {
    val layer = classOf[Dropout[T]].cast(module)
    layer.initP
  }

  protected def toCaffeBatchNormParam(module : AbstractModule[Activity, Tensor[T], T]) : Double = {
    val layer = classOf[BatchNormalization[T]].cast(module)
    layer.eps
  }

  protected def toCaffeConcatParam(module : AbstractModule[Activity, Tensor[T], T]) : Int = {
    val layer = classOf[JoinTable[T]].cast(module)
    layer.dimension
  }

  protected def toCaffeEluParam(module : AbstractModule[Activity, Tensor[T], T]) : ELUParameter = {
    val eLUParameter = ELUParameter.newBuilder()
    val layer = classOf[ELU[T]].cast(module)
    eLUParameter.setAlpha(layer.alpha.toFloat)
    eLUParameter.build()
  }

  protected def toCaffePowerParam(module : AbstractModule[Activity, Tensor[T], T])
    : PowerParameter = {
    val powerParameter = PowerParameter.newBuilder
    val layer = classOf[Power[T]].cast(module)
    powerParameter.setPower(layer.power.toFloat)
    powerParameter.setScale(layer.scale.toFloat)
    powerParameter.setShift(layer.shift.toFloat)
    powerParameter.build
  }

  protected def toCaffeReshapeParam(module : AbstractModule[Activity, Tensor[T], T])
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

  protected def toCaffeScalaParam(module : AbstractModule[Activity, Tensor[T], T]) : BlobShape = {
    val layer = classOf[Scale[T]].cast(module)
    val size = layer.size
    val shapeBlob = BlobShape.newBuilder
    var i = 0
    while (i < size.length) {
      shapeBlob.setDim(i, size(i))
    }
    shapeBlob.build
  }

  protected def toCaffeThresholdParam(module : AbstractModule[Activity, Tensor[T], T])
    : ThresholdParameter = {
    val layer = classOf[Threshold[T]].cast(module)
    val threshold = layer.threshold
    val thresholdParameter = ThresholdParameter.newBuilder
    thresholdParameter.setThreshold(threshold.toFloat)
    thresholdParameter.build
  }

  protected def toCaffeSliceParam(module : AbstractModule[Activity, Tensor[T], T])
    : SliceParameter = {
    val layer = classOf[SplitTable[T]].cast(module)
    val axis = layer.dimension
    val sliceParameter = SliceParameter.newBuilder
    sliceParameter.setAxis(axis)
    sliceParameter.build
  }

  protected def toCaffeTileParam(module : AbstractModule[Activity, Tensor[T], T])
    : TileParameter = {
    val layer = classOf[Replicate[T]].cast(module)
    val tile = layer.nFeatures
    val axis = layer.dim
    val tileParameter = TileParameter.newBuilder
    tileParameter.setTiles(tile)
    tileParameter.setAxis(axis)
    tileParameter.build
  }

}
