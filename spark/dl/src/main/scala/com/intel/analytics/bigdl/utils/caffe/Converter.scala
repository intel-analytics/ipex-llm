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

import caffe.Caffe._
import caffe.Caffe.EltwiseParameter.EltwiseOp
import caffe.Caffe.PoolingParameter.PoolMethod
import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable
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
    throw new UnsupportedOperationException(s"$layerType is not supported in BigDL for now")
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
    Seq(ReLU(true).setName(layerName).apply())
  }

  private def fromCaffeLRN(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    val param = getLRNParam(layer).get
    val localSize = param.getLocalSize
    val alpha = param.getAlpha
    val belta = param.getBeta
    val k = param.getK
    Seq(SpatialCrossMapLRN[T](localSize, alpha, belta, k).setName(layerName).apply())
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
        setName(layerName).apply()
      case PoolMethod.AVE => SpatialAveragePooling[T](kw, kh, dw, dh, pw, ph).ceil().
        setName(layerName).apply()
      case _ => null
    }
    Seq(pooling)
  }

  private def fromCaffeDropout(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = getDropoutParam(layer).get
    val layerName = getLayerName(layer)
    val initP = param.getDropoutRatio
    Seq(Dropout[T](initP).setName(layerName).apply())
  }

  private def fromCaffeSoftmax(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(LogSoftMax().setName(layerName).apply())
  }

  private def fromCaffeTanh(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(Tanh[T]().setName(layerName).apply())
  }

  private def fromCaffeSigmoid(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(Sigmoid[T]().setName(layerName).apply())
  }

  private def fromCaffeAbsVal(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(Abs[T]().setName(layerName).apply())
  }

  private def fromCaffeConcat(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    val param = getConcatParam(layer)
    val dim = param.get.getAxis
    Seq(JoinTable[T](dim + 1, 0).setName(layerName).apply())
  }

  private def fromCaffeFlatten(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(FlattenTable[T].setName(layerName).apply())
  }

  private def fromCaffeLog(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(Log[T]().setName(layerName).apply())
  }

  private def fromCaffePower(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    val param = getPowerParam(layer).get
    val power = param.getPower
    var scale = 1.0
    var shift = 0.0
    if (param.hasScale) scale = param.getScale
    if (param.hasShift) shift = param.getShift
    Seq(Power[T](power, scale, shift).setName(layerName).apply())
  }

  private def fromCaffePreLU(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(PReLU[T]().setName(layerName).apply())
  }

  private def fromCaffeRecurrent(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(Recurrent[T]().setName(layerName).apply())
  }

  private def fromCaffeThreshold(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = getThresholdParam(layer).get
    var threshold = 1e-6
    if (param.hasThreshold) {
      threshold = param.getThreshold
    }
    Seq(Threshold[T](threshold).setName(getLayerName(layer)).apply())
  }

  private def fromCaffeExp(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val layerName = getLayerName(layer)
    Seq(Exp[T]().setName(layerName).apply())
  }

  private def fromCaffeSlice(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = getSliceParam(layer)
    val layerName = getLayerName(layer)
    val axis = param.get.getAxis
    Seq(SplitTable[T](axis).setName(layerName).apply())
  }

  private def fromCaffeEltwise(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = getEltWiseParam(layer).get
    val layerName = getLayerName(layer)
    val opsType = param.getOperation
    val coeff2 = param.getCoeff(1)
    val ops = opsType match {
      case EltwiseOp.PROD => CMulTable[T]().setName(layerName).apply()
      case EltwiseOp.MAX => CMaxTable[T]().setName(layerName).apply()
      case EltwiseOp.SUM =>
        if (coeff2 < 0) {
          CAddTable[T]().setName(layerName).apply()
        } else {
          CSubTable[T]().setName(layerName).apply()
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

  private def init() = {
    caffe2BigDL("CONVOLUTION") = fromCaffeConvolution
    caffe2BigDL("INNERPRODUCT") = fromCaffeInnerProduct
    caffe2BigDL("INNER_PRODUCT") = fromCaffeInnerProduct
    caffe2BigDL("RELU") = fromCaffeReLU
    caffe2BigDL("LRN") = fromCaffeLRN
    caffe2BigDL("POOLING") = fromCaffePooling
    caffe2BigDL("DROPOUT") = fromCaffeDropout
    caffe2BigDL("SOFTMAX") = fromCaffeSoftmax
    caffe2BigDL("SOFTMAX_LOSS") = fromCaffeSoftmax
    caffe2BigDL("SOFTMAXWITHLOSS") = fromCaffeSoftmax
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
    caffe2BigDL("INPUT") = null
    caffe2BigDL("DATA") = null
  }
}
