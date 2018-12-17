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

import scala.collection.JavaConverters._
import caffe.Caffe
import caffe.Caffe.EltwiseParameter.EltwiseOp
import caffe.Caffe.LRNParameter.NormRegion
import caffe.Caffe.V1LayerParameter.LayerType
import caffe.Caffe._
import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Node, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * [[Converter]] implementation for caffe deprecated LayerParameter conversion
 */
class V1LayerConverter[T: ClassTag](implicit ev: TensorNumeric[T]) extends Converter[T] {

  override protected def fromCaffeConvolution(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = getConvolutionParam(layer).get
    val weightBlob = getBlob(layer, 0)
    sanityBlobCheck(layer, "weight", weightBlob)
    val weight = weightBlob.get
    val biasBlob = getBlob(layer, 1)
    val withBias = biasBlob.isDefined
    val group = if (param.getGroup == 0)  1 else param.getGroup
    val channel = if (weight.getShape.getDimCount > 1) weight.getShape.getDim(1).toInt
    else weight.getChannels
    val nInputPlane = channel * group
    val nOutPlane = param.getNumOutput
    var kw = param.getKernelW
    var kh = param.getKernelH
    var dw = param.getStrideW
    var dh = param.getStrideH
    if (kw ==0 || kh == 0) {
      kw = param.getKernelSize(0)
      kh = kw
    }
    if (dw == 0 || dh == 0) {
      if (param.getStrideList.size() != 0) {
        dw = param.getStride(0)
        dh = dw
      } else {
        // use default values if not found
        dw = 1
        dh = 1
      }
    }
    var pw = param.getPadW
    var ph = param.getPadH
    if (pw == 0 || ph == 0) {
      if (param.getPadList.size() != 0) {
        pw = param.getPad(0)
        ph = pw
      }
    }
    val layerType = getLayerType(layer).toUpperCase
    if ("DECONVOLUTION" == layerType) {
      Seq(SpatialFullConvolution[T](nOutPlane, nInputPlane, kw, kh, dw, dh, pw, ph, 0, 0, group,
      noBias = !withBias)
        .setName(getLayerName(layer)).inputs())
    } else {
      Seq(SpatialConvolution[T](nInputPlane, nOutPlane, kw, kh, dw, dh, pw, ph, group,
      withBias = withBias)
        .setName(getLayerName(layer)).inputs())
    }
  }

  override protected def fromCaffeInnerProduct(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = getInnerProductParam(layer).get
    val withBias = param.getBiasTerm
    val layerName = getLayerName(layer)
    val weightBlob = getBlob(layer.asInstanceOf[V1LayerParameter], 0)
    sanityBlobCheck(layer, "weight", weightBlob)
    val weight = weightBlob.get
    val nInputPlane = if (weight.getShape.getDimCount > 1) weight.getShape.getDim(1).toInt
    else weight.getWidth
    val nOutputPlane = param.getNumOutput
    val linear = Linear[T](nInputPlane, nOutputPlane, withBias = withBias).setName(layerName)
    val node = linear.inputs()
    if(nInputPlane != nOutputPlane) {
      // Construct a view layer in between
      val view = View[T](nInputPlane).inputs()
      view -> node
      Seq(view, node)
    } else {
      Seq(node)
    }
  }

  // No implementation in V1
  override protected def fromCaffeBatchNormalization(layer : GeneratedMessage) :
    Seq[ModuleNode[T]] = {
    throw new CaffeConversionException("Batch normalizaton is not supported in V1 Layer")
  }

  // No implementation in V1
  override protected def fromCaffeELU(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    throw new CaffeConversionException("ELU is not supported in V1 Layer")
  }

  // No implementation in V1
  override protected def fromCaffeReshape(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    throw new CaffeConversionException("Reshape is not supported in V1 Layer")
  }

  // No implementation in V1
  override protected def fromCaffeScale(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    throw new CaffeConversionException("Scale is not supported in V1 Layer")
  }

  // No implementation in V1
  override protected def fromCaffeBias(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    throw new CaffeConversionException("Bias is not supported in V1 Layer")
  }

  // No implementation in V1
  override protected def fromCaffeTile(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    throw new CaffeConversionException("Tile is not supported in V1 Layer")
  }

  override protected def fromCaffeInput(layer: GeneratedMessage): Seq[ModuleNode[T]] = {
    val tops = layer.asInstanceOf[V1LayerParameter].getTopList
    (0 until tops.size()).map(i => {
      val input = Input()
      input.element.setName(tops.get(i))
      input
    })
  }

  override protected def toCaffeConvolution(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = V1LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    layerParameter.setType(LayerType.CONVOLUTION)

    // set bottom list and top list
    setConnections(layerParameter, bottoms, nextSize)

    // copy weight and bias
    val (weightBuilder, biasBuilder) = copyParam(module)

    // get convolution param map
    val layerParams = toCaffeConvolutionParam(module)

    val convolutionParam = ConvolutionParameter.newBuilder()

    val ngroup = layerParams("ngroup")
    val nInputPlane = layerParams("nInputPlane")
    val nOutputPlane = layerParams("nOutputPlane")
    convolutionParam.setGroup(ngroup)
    convolutionParam.setNumOutput(nOutputPlane)
    convolutionParam.setKernelW(layerParams("kernelW"))
    convolutionParam.setKernelH(layerParams("kernelH"))
    convolutionParam.setStrideW(layerParams("strideW"))
    convolutionParam.setStrideH(layerParams("strideH"))
    convolutionParam.setPadW(layerParams("padW"))
    convolutionParam.setPadH(layerParams("padH"))
    val withBias = if (layerParams("withBias") == 1) true else false
    convolutionParam.setBiasTerm(withBias)

    weightBuilder.setChannels(nInputPlane / ngroup)
    weightBuilder.setNum(nOutputPlane)

    setBlobs(layerParameter, weightBuilder, biasBuilder)

    layerParameter.setConvolutionParam(convolutionParam.build)

    // build concolution layer
    Seq(layerParameter.build())

  }

  override protected def toCaffeDeConvolution(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {

    val layerParameter = V1LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    layerParameter.setType(LayerType.DECONVOLUTION)

    // set bottom list and top list
    setConnections(layerParameter, bottoms, nextSize)

    // copy weight and bias
    val (weightBuilder, biasBuilder) = copyParam(module)

    // get deconvolution param map
    val layerParams = toCaffeDeConvolutionParam(module)

    val convolutionParam = ConvolutionParameter.newBuilder()

    val ngroup = layerParams("ngroup")
    val nInputPlane = layerParams("nInputPlane")
    val nOutputPlane = layerParams("nOutputPlane")
    convolutionParam.setGroup(ngroup)
    convolutionParam.setNumOutput(nOutputPlane)
    convolutionParam.setKernelW(layerParams("kernelW"))
    convolutionParam.setKernelH(layerParams("kernelH"))
    convolutionParam.setStrideW(layerParams("strideW"))
    convolutionParam.setStrideH(layerParams("strideH"))
    convolutionParam.setPadW(layerParams("padW"))
    convolutionParam.setPadH(layerParams("padH"))
    val withBias = if (layerParams("withBias") == 1) true else false
    convolutionParam.setBiasTerm(withBias)

    weightBuilder.setChannels(nInputPlane / ngroup)
    weightBuilder.setNum(nOutputPlane)

    setBlobs(layerParameter, weightBuilder, biasBuilder)

    layerParameter.setConvolutionParam(convolutionParam.build)

    // build deConcolution layer
    Seq(layerParameter.build())
  }

  override protected def toCaffeRelu(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {

    val layerParameter = V1LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    layerParameter.setType(LayerType.RELU)

    // set bottom list and top list
    setConnections(layerParameter, bottoms, nextSize)

    // copy weight and bias
    val (weightBuilder, biasBuilder) = copyParam(module)
    setBlobs(layerParameter, weightBuilder, biasBuilder)
    // build concolution layer
    Seq(layerParameter.build())
  }

  override protected def toCaffeLRN(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = V1LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    layerParameter.setType(LayerType.LRN)

    // set bottom list and top list
    setConnections(layerParameter, bottoms, nextSize)

    // copy weight and bias
    val (weightBuilder, biasBuilder) = copyParam(module)

    val (localSize, alpha, belta, k, lrnType) = toCaffeLRNParam(module)

    val lrnParameter = LRNParameter.newBuilder()

    lrnParameter.setLocalSize(localSize)
    lrnParameter.setAlpha(alpha.toFloat)
    lrnParameter.setBeta(belta.toFloat)
    lrnParameter.setK(k.toFloat)
    if (lrnType == SpatialCrossMapLRN.getClass.getSimpleName) {
      lrnParameter.setNormRegion(NormRegion.ACROSS_CHANNELS)
    } else if (lrnType == SpatialWithinChannelLRN.getClass.getSimpleName) {
      lrnParameter.setNormRegion(NormRegion.WITHIN_CHANNEL)
    }
    setBlobs(layerParameter, weightBuilder, biasBuilder)

    layerParameter.setLrnParam(lrnParameter.build)

    Seq(layerParameter.build)
  }

  override protected def toCaffeMaxPooling(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    toCaffePooling(module, bottoms, nextSize, true)
  }

  override protected def toCaffeAvePooling(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    toCaffePooling(module, bottoms, nextSize, false)
  }

  private def toCaffePooling(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int, max : Boolean): Seq[GeneratedMessage] = {
    val layerParameter = V1LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    layerParameter.setType(LayerType.POOLING)

    // set bottom list and top list
    setConnections(layerParameter, bottoms, nextSize)

    // copy weight and bias
    val (weightBuilder, biasBuilder) = copyParam(module)

    val poolingParameter = if (max) toCaffeMaxPoolingParam(module)
    else toCaffeAvgPoolingParam(module)

    setBlobs(layerParameter, weightBuilder, biasBuilder)

    layerParameter.setPoolingParam(poolingParameter)

    Seq(layerParameter.build)
  }

  override protected def toCaffeInnerProduct(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = V1LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    layerParameter.setType(LayerType.INNER_PRODUCT)

    // set bottom list and top list
    setConnections(layerParameter, bottoms, nextSize)

    // copy weight and bias
    val (weightBuilder, biasBuilder) = copyParam(module)

    val (inputSize, outputSize, withBias) = toCaffeInnerProductParam(module)

    weightBuilder.setWidth(inputSize)

    val innerProductParameter = InnerProductParameter.newBuilder

    innerProductParameter.setNumOutput(outputSize)

    innerProductParameter.setBiasTerm(withBias)

    setBlobs(layerParameter, weightBuilder, biasBuilder)

    layerParameter.setInnerProductParam(innerProductParameter.build)

    Seq(layerParameter.build)
  }

  override protected def toCaffeDropOut(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).
      setType(LayerType.DROPOUT)

    val dropOutRatio = toCaffeDropOutParam(module)

    val dropoutParameter = DropoutParameter.newBuilder

    dropoutParameter.setDropoutRatio(dropOutRatio.toFloat)

    layerParameter.setDropoutParam(dropoutParameter.build)

    Seq(layerParameter.build)

  }

  override protected def toCaffeLogSoftMax(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {

    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType(LayerType.SOFTMAX).build)
  }

  override protected def toCaffeTanh(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {

    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType(LayerType.TANH).build)
  }

  override protected def toCaffeSigmoid(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType(LayerType.SIGMOID).build)
  }

  override protected def toCaffeAbs(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType(LayerType.ABSVAL).build)
  }

  override protected def toCaffeBatchNormalization(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    throw new CaffeConversionException("Batch normalization is not supported in V1Layer")
  }

  override protected def toCaffeConcat(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).
      setType(LayerType.CONCAT)
    val dimension = toCaffeConcatParam(module)
    val concatParameter = ConcatParameter.newBuilder
    concatParameter.setAxis(dimension - 1)
    layerParameter.setConcatParam(concatParameter.build)
    Seq(layerParameter.build)
  }

  override protected def toCaffeElu(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    throw new CaffeConversionException("ELU is not supported in V1Layer")
  }

  override protected def toCaffeFlattern(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType(LayerType.FLATTEN).build)
  }

  override protected def toCaffeLog(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    throw new CaffeConversionException("LOG is not supported in V1Layer")
  }

  override protected def toCaffePower(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).
      setType(LayerType.POWER)
    layerParameter.setPowerParam(toCaffePowerParam(module))
    Seq(layerParameter.build)
  }

  override protected def toCaffePReLu(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    throw new CaffeConversionException("PReLU is not supported in V1Layer")
  }

  override  protected def toCaffeRecurrent(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    throw new CaffeConversionException("Recurrent is not supported in V1Layer")
  }

  override protected def toCaffeReshape(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    throw new CaffeConversionException("Reshape is not supported in V1Layer")
  }

  override protected def toCaffeScale(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    throw new CaffeConversionException("Scale is not supported in V1Layer")
  }

  override protected def toCaffeBias(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    throw new CaffeConversionException("Bias is not supported in V1Layer")
  }

  override  protected def toCaffeThreshold(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).
      setType(LayerType.THRESHOLD)
    val thresholdParameter = toCaffeThresholdParam(module)
    layerParameter.setThresholdParam(thresholdParameter)
    Seq(layerParameter.build)
  }

  override protected def toCaffeExp(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType(LayerType.EXP).build)
  }

  override protected def toCaffeSlice(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).
      setType(LayerType.SLICE)
    val sliceParameter = toCaffeSliceParam(module)
    layerParameter.setSliceParam(sliceParameter)
    Seq(layerParameter.build)
  }

  override protected def toCaffeTile(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    throw new CaffeConversionException("Tile is not supported in V1Layer")
  }

  override protected def toCaffeEltWiseMax(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize)
      .setType(LayerType.ELTWISE)
    val eltwiseParameter = EltwiseParameter.newBuilder
    eltwiseParameter.setOperation(EltwiseOp.MAX)
    layerParameter.setEltwiseParam(eltwiseParameter)
    Seq(layerParameter.build)
  }

  override protected def toCaffeEltWiseAdd(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).
      setType(LayerType.ELTWISE)
    val eltwiseParameter = EltwiseParameter.newBuilder
    eltwiseParameter.setOperation(EltwiseOp.SUM)
    eltwiseParameter.setCoeff(1, 1.0f)
    layerParameter.setEltwiseParam(eltwiseParameter)
    Seq(layerParameter.build)
  }

  override protected def toCaffeEltWiseSub(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).
      setType(LayerType.ELTWISE)
    val eltwiseParameter = EltwiseParameter.newBuilder
    eltwiseParameter.setOperation(EltwiseOp.SUM)
    eltwiseParameter.setCoeff(1, -1.0f)
    layerParameter.setEltwiseParam(eltwiseParameter)
    Seq(layerParameter.build)
  }

  override protected def toCaffeSequential(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val res = new ArrayBuffer[GeneratedMessage]()
    var lastBottoms = bottoms
    val modules = module.asInstanceOf[Sequential[T]].modules
    modules.foreach(nested => {
      val nestedLayer = nested.asInstanceOf[AbstractModule[_, _, _]].
        asInstanceOf[AbstractModule[Activity, Activity, T]]
      val nextedLayers = toCaffe(nestedLayer, lastBottoms, nextSize)
      res.appendAll(nextedLayers)
      lastBottoms.clear()
      nextedLayers(nextedLayers.size - 1).
        asInstanceOf[V1LayerParameter].getTopList.asScala.foreach(lastBottoms.append(_))
    })
    res
  }

  private def toCaffeWithWeightAndBiasOnly(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): V1LayerParameter.Builder = {

    val layerParameter = V1LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    // set bottom list and top list
    setConnections(layerParameter, bottoms, nextSize)

    // copy weight and bias
    val (weightBuilder, biasBuilder) = copyParam(module)

    setBlobs(layerParameter, weightBuilder, biasBuilder)

    layerParameter
  }

  private def setConnections(layerParameter : V1LayerParameter.Builder,
                             bottoms : ArrayBuffer[String], nextSize : Int) : Unit = {

    val layerName = layerParameter.getName

    // set bottom list
    var i = 0
    bottoms.foreach(bottom => {
      layerParameter.addBottom(bottom)
      i += 1
    })

    // set top list
    i = 0
    while (i < nextSize) {
      layerParameter.addTop(s"$layerName$i")
      i += 1
    }
  }

  private def setBlobs(layerParameterBuilder: V1LayerParameter.Builder,
                       blobs : BlobProto.Builder*) : Unit = {
    blobs.foreach(blobBuilder => {
      if (blobBuilder != null) {
        layerParameterBuilder.addBlobs(blobBuilder.build)
      }
    })
  }

  private def copyParam(module : AbstractModule[Activity, Activity, T]) :
  (BlobProto.Builder, BlobProto.Builder) = {
    // weight and bias may be empty
    var weightBlobBuilder : BlobProto.Builder = null
    var biasBlobBuilder : BlobProto.Builder = null
    val name = module.getName
    if (module.getParametersTable != null) {
      val params = module.getParametersTable.get(name).get.asInstanceOf[Table]
      if (params.contains("weight")) {
        weightBlobBuilder = BlobProto.newBuilder()
        val weight = params[Tensor[T]]("weight")
        val weightData = weight.storage().array()
        var i = 0
        while (i < weightData.length) {
          weightBlobBuilder.addData(ev.toType[Float](weightData(i)))
          i += 1
        }
      }
      if (params.contains("bias")) {
        biasBlobBuilder = BlobProto.newBuilder()
        val bias = params[Tensor[T]]("bias")
        val biasData = bias.storage().array()
        var i = 0
        while (i < biasData.length) {
          biasBlobBuilder.addData(ev.toType[Float](biasData(i)))
          i += 1
        }
      }
    }
    (weightBlobBuilder, biasBlobBuilder)
  }

  override protected def getLayerName(layer : GeneratedMessage) : String = {
    layer.asInstanceOf[V1LayerParameter].getName
  }

  override protected def getLayerType(layer : GeneratedMessage) : String = {
    layer.asInstanceOf[V1LayerParameter].getType.toString
  }

  protected def getConvolutionParam(layer : GeneratedMessage): Option[ConvolutionParameter] = {
    Some(layer.asInstanceOf[V1LayerParameter].getConvolutionParam)
  }

  override protected def getLRNParam(layer: GeneratedMessage): Option[LRNParameter] = {
    Some(layer.asInstanceOf[V1LayerParameter].getLrnParam)
  }

  override protected def getPoolingParam(layer : GeneratedMessage): Option[PoolingParameter] = {
    Some(layer.asInstanceOf[V1LayerParameter].getPoolingParam)
  }

  override protected def getInnerProductParam(layer : GeneratedMessage):
    Option[InnerProductParameter] = {
    Some(layer.asInstanceOf[V1LayerParameter].getInnerProductParam)
  }

  override protected def getDropoutParam(layer : GeneratedMessage): Option[DropoutParameter] = {
    Some(layer.asInstanceOf[V1LayerParameter].getDropoutParam)
  }

  override protected def getConcatParam(layer : GeneratedMessage): Option[ConcatParameter] = {
    Some(layer.asInstanceOf[V1LayerParameter].getConcatParam)
  }

  override protected def getPowerParam(layer : GeneratedMessage) : Option[PowerParameter] = {
    Some(layer.asInstanceOf[V1LayerParameter].getPowerParam)
  }

  override protected def getThresholdParam(layer : GeneratedMessage): Option[ThresholdParameter] = {
    Some(layer.asInstanceOf[V1LayerParameter].getThresholdParam)
  }

  override protected def getSliceParam(layer : GeneratedMessage): Option[SliceParameter] = {
    Some(layer.asInstanceOf[V1LayerParameter].getSliceParam)
  }

  override protected def getEltWiseParam(layer : GeneratedMessage): Option[EltwiseParameter] = {
    Some(layer.asInstanceOf[V1LayerParameter].getEltwiseParam)
  }

  protected def getBlob(layer : GeneratedMessage, ind: Int): Option[Caffe.BlobProto] = {
    if (layer.asInstanceOf[V1LayerParameter].getBlobsCount > ind) {
      Some(layer.asInstanceOf[V1LayerParameter].getBlobs(ind))
    } else {
      None
    }
  }
}
