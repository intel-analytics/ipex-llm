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
import caffe.Caffe.{BlobProto, PoolingParameter, _}
import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 *  [[Converter]] implementation for caffe recommended LayerParameter conversion
 */
class LayerConverter[T: ClassTag](implicit ev: TensorNumeric[T]) extends Converter[T]{

  override protected def fromCaffeConvolution(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = getConvolutionParam(layer).get
    val group = if (param.getGroup == 0)  1 else param.getGroup
    var  weightBlob = getBlob(layer, 0)
    sanityBlobCheck(layer, "weight", weightBlob)
    val weight = weightBlob.get
    val biasBlob = getBlob(layer, 1)
    val withBias = biasBlob.isDefined
    val nInputPlane = if (weight.hasShape) weight.getShape.getDim(1) * group
    else weight.getChannels * group
    val nOutPlane = if (weight.hasShape) weight.getShape.getDim(0)
    else weight.getNum
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

    if (param.getDilationCount == 0 || param.getDilation(0) == 1) {
      val layerType = getLayerType(layer).toUpperCase
      if ("DECONVOLUTION" == layerType) {
        Seq(SpatialFullConvolution[T](nOutPlane.toInt, nInputPlane.toInt,
          kw, kh, dw, dh, pw, ph, 0, 0, group, !withBias).setName(getLayerName(layer)).inputs())
      } else {
        Seq(SpatialConvolution[T](nInputPlane.toInt, nOutPlane.toInt,
          kw, kh, dw, dh, pw, ph, group, withBias = withBias).setName(getLayerName(layer)).inputs())
      }
    } else {
      val dilation = param.getDilation(0)
      Seq(SpatialDilatedConvolution[T](nInputPlane.toInt, nOutPlane.toInt,
        kw, kh, dw, dh, pw, ph, dilation, dilation)
        .setName(getLayerName(layer)).inputs())
    }
  }

  override protected def fromCaffeInnerProduct(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = getInnerProductParam(layer).get
    val withBias = param.getBiasTerm
    val layerName = getLayerName(layer)
    val weightBlob = getBlob(layer.asInstanceOf[LayerParameter], 0)
    sanityBlobCheck(layer, "weight", weightBlob)
    val weight = weightBlob.get
    var nInputPlane = 0
    if (weight.hasShape) {
      nInputPlane = weight.getShape.getDim(1).toInt
    }
    else {
      nInputPlane = weight.getWidth
    }
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

  override protected def fromCaffeBatchNormalization(layer: GeneratedMessage):
  Seq[ModuleNode[T]] = {
    val weightBlob = getBlob(layer, 0)
    sanityBlobCheck(layer, "weight", weightBlob)
    val weight = weightBlob.get
    val nOutPlane = if (weight.hasShape) weight.getShape.getDim(0).toInt
    else weight.getNum
    val param = layer.asInstanceOf[LayerParameter].getBatchNormParam
    val eps = param.getEps
    val batchNorm = SpatialBatchNormalization[T](nOutPlane.toInt, eps, affine = false)
        .setName(getLayerName(layer))
    val scalaBlob = getBlob(layer, 2)
    sanityBlobCheck(layer, "scale", scalaBlob)
    val scaleData = scalaBlob.get.getData(0)
    val scale = if (scaleData == 0) 0 else 1 / scaleData
    sanityBlobCheck(layer, "mean", getBlob(layer, 0))
    val means = getBlob(layer, 0).get.getDataList
    sanityBlobCheck(layer, "variance", getBlob(layer, 1))
    val variances = getBlob(layer, 1).get.getDataList
    batchNorm.runningMean.resize(nOutPlane)
    batchNorm.runningVar.resize(nOutPlane)

    batchNorm.saveMean.resize(nOutPlane)
    batchNorm.saveStd.resize(nOutPlane)
    (1 to nOutPlane).foreach(i => {
      batchNorm.runningMean.setValue(i, ev.fromType[Float](means.get(i - 1) * scale))
      batchNorm.runningVar.setValue(i, ev.fromType[Float](variances.get(i - 1) * scale))
    })
    Seq(batchNorm.inputs())
  }

  override protected def fromCaffeELU(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = layer.asInstanceOf[LayerParameter].getEluParam
    var alpha = 1.0
    if (param.hasAlpha) alpha = param.getAlpha
    Seq(ELU[T](alpha).setName(getLayerName(layer)).inputs())
  }

  override protected def fromCaffeReshape(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = layer.asInstanceOf[LayerParameter].getReshapeParam
    val shapeSize = param.getShape.getDimList.asScala.map(_.toInt).toArray
    Seq(InferReshape[T](shapeSize).setName(getLayerName(layer)).inputs())
  }

  override protected def fromCaffeScale(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = layer.asInstanceOf[LayerParameter].getScaleParam
    val layerName = getLayerName(layer)
    // second blob as weight for scale
    val weightBlob = getBlob(layer, 1)
    if (weightBlob.isDefined) {
      val blob = weightBlob.get
      val size : Array[Int] = if (blob.getShape.getDimCount == 1) {
        Array(1, blob.getShape.getDim(0).toInt, 1, 1)
      } else {
        blob.getShape.getDimList.asScala.map(_.toInt).toArray
      }
      Seq(Scale[T](size).setName(layerName).inputs())
    } else {
      val inputBlob = getBlob(layer, 0)
      sanityBlobCheck(layer, "weight", getBlob(layer, 0))
      val input = inputBlob.get
      val shape = input.getShape
      val axis = param.getAxis
      var numOfAxis = param.getNumAxes
      if (numOfAxis == -1) {
        numOfAxis = shape.getDimList.size() - 1
      } else {
        numOfAxis = numOfAxis + axis
      }
      val size = shape.getDimList.subList(axis - 1, numOfAxis - 1).asScala.map(_.toInt).toArray
      Seq(CMul[T](size).setName(layerName).inputs())
    }
  }

  override protected def fromCaffeBias(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    // input blob
    val weightBlob = getBlob(layer, 0)
    sanityBlobCheck(layer, "weight", weightBlob)
    val size = weightBlob.get.getShape.getDimList.asScala.map(_.toInt).toArray.product
    Seq(Add[T](size).setName(getLayerName(layer)).inputs())
  }

  override protected def fromCaffeTile(layer : GeneratedMessage) : Seq[ModuleNode[T]] = {
    val param = layer.asInstanceOf[LayerParameter].getTileParam
    val axis = param.getAxis
    val tiles = param.getTiles
    Seq(Tile[T](axis + 1, tiles).setName(getLayerName(layer)).inputs())

  }

  override protected def fromCaffeInput(layer: GeneratedMessage): Seq[ModuleNode[T]] = {
    val layerParam = layer.asInstanceOf[LayerParameter]
    val tops = layerParam.getTopList
    (0 until tops.size()).map(i => {
      val input = Input()
      input.element.setName(tops.get(i))
      input
    })
  }

  override protected def toCaffeConvolution(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    layerParameter.setType("Convolution")

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
    val layerParameter = LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    layerParameter.setType("Deconvolution")

    // set bottom list and top list
    setConnections(layerParameter, bottoms, nextSize)

    // copy weight and bias
    val (weightBuilder, biasBuilder) = copyParam(module)

    // get convolution param map
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

    // build concolution layer
    Seq(layerParameter.build())
  }
  override protected def toCaffeRelu(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {

    val layerParameter = LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    layerParameter.setType("ReLU")

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
    val layerParameter = LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    layerParameter.setType("LRN")

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
    val layerParameter = LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    layerParameter.setType("Pooling")

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
    val layerParameter = LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    layerParameter.setType("InnerProduct")

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

    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("Dropout")

    val dropOutRatio = toCaffeDropOutParam(module)

    val dropoutParameter = DropoutParameter.newBuilder

    dropoutParameter.setDropoutRatio(dropOutRatio.toFloat)

    layerParameter.setDropoutParam(dropoutParameter.build)

    Seq(layerParameter.build)
  }

  override protected def toCaffeLogSoftMax(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {

    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("Softmax").build)
  }

  override protected def toCaffeTanh(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {

    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("TanH").build)
  }

  override protected def toCaffeSigmoid(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("Sigmoid").build)
  }

  override protected def toCaffeAbs(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("AbsVal").build)
  }

  override protected def toCaffeBatchNormalization(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).
      setType("BatchNorm")
    val batchNormParameter = BatchNormParameter.newBuilder()
    val eps = toCaffeBatchNormParam(module)
    batchNormParameter.setEps(eps.toFloat)
    layerParameter.setBatchNormParam(batchNormParameter.build)
    Seq(layerParameter.build)
  }

  override protected def toCaffeConcat(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("Concat")
    val dimension = toCaffeConcatParam(module)
    val concatParameter = ConcatParameter.newBuilder
    concatParameter.setAxis(dimension - 1)
    layerParameter.setConcatParam(concatParameter.build)
    Seq(layerParameter.build)
  }

  override protected def toCaffeElu(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).
      setType("ELU")
    layerParameter.setEluParam(toCaffeEluParam(module))
    Seq(layerParameter.build)
  }

  override protected def toCaffeFlattern(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("Flatten").build)
  }

  override protected def toCaffeLog(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("Log").build)
  }

  override protected def toCaffePower(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("Power")
    layerParameter.setPowerParam(toCaffePowerParam(module))
    Seq(layerParameter.build)
  }

  override protected def toCaffePReLu(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("PReLU").build)
  }

  override protected def toCaffeRecurrent(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("Recurrent").build)
  }

  override protected def toCaffeReshape(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("Reshape")
    layerParameter.setReshapeParam(toCaffeReshapeParam(module))
    Seq(layerParameter.build)
  }

  override protected def toCaffeScale(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    layerParameter.setType("Scale")

    // set bottom list and top list
    setConnections(layerParameter, bottoms, nextSize)

    // copy weight and bias
    val (weightBuilder, biasBuilder) = copyParam(module)

    val blobShape = toCaffeScalaParam(module)

    biasBuilder.setShape(blobShape)

    setBlobs(layerParameter, weightBuilder, biasBuilder)

    Seq(layerParameter.build)
  }

  override protected def toCaffeBias(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("Bias").build)
  }

  override  protected def toCaffeThreshold(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).
      setType("Threshold")
    val thresholdParameter = toCaffeThresholdParam(module)
    layerParameter.setThresholdParam(thresholdParameter)
    Seq(layerParameter.build)
  }

  override protected def toCaffeExp(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    Seq(toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("Exp").build)
  }

  override protected def toCaffeSlice(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("Slice")
    val sliceParameter = toCaffeSliceParam(module)
    layerParameter.setSliceParam(sliceParameter)
    Seq(layerParameter.build)
  }

  override protected def toCaffeTile(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("Tile")
    val tileParameter = toCaffeTileParam(module)
    layerParameter.setTileParam(tileParameter.toBuilder)
    Seq(layerParameter.build)
  }

  override protected def toCaffeEltWiseMax(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("EltWise")
    val eltwiseParameter = EltwiseParameter.newBuilder
    eltwiseParameter.setOperation(EltwiseOp.MAX)
    layerParameter.setEltwiseParam(eltwiseParameter)
    Seq(layerParameter.build)
  }

  override protected def toCaffeEltWiseAdd(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("EltWise")
    val eltwiseParameter = EltwiseParameter.newBuilder
    eltwiseParameter.setOperation(EltwiseOp.SUM)
    eltwiseParameter.setCoeff(1, 1.0f)
    layerParameter.setEltwiseParam(eltwiseParameter)
    Seq(layerParameter.build)
  }

  override protected def toCaffeEltWiseSub(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(module, bottoms, nextSize).setType("EltWise")
    val eltwiseParameter = EltwiseParameter.newBuilder
    eltwiseParameter.setOperation(EltwiseOp.SUM)
    eltwiseParameter.setCoeff(1, -1.0f)
    layerParameter.setEltwiseParam(eltwiseParameter)
    Seq(layerParameter.build)
  }

  override protected def toCaffeSequential(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): Seq[GeneratedMessage] = {
    val res = new ArrayBuffer[GeneratedMessage]()
    val lastBottoms = bottoms
    val modules = module.asInstanceOf[Sequential[T]].modules
    modules.foreach(nested => {
      val nestedLayer = nested.asInstanceOf[AbstractModule[_, _, _]].
        asInstanceOf[AbstractModule[Activity, Activity, T]]
      val nextedLayers = toCaffe(nestedLayer, lastBottoms, nextSize)
      res.appendAll(nextedLayers)
      lastBottoms.clear()
      nextedLayers(nextedLayers.size - 1).
        asInstanceOf[LayerParameter].getTopList.asScala.foreach(lastBottoms.append(_))
    })
    res
  }

  private def toCaffeWithWeightAndBiasOnly(module : AbstractModule[Activity, Activity, T],
    bottoms : ArrayBuffer[String], nextSize : Int): LayerParameter.Builder = {

    val layerParameter = LayerParameter.newBuilder()

    val layerName = module.getName

    layerParameter.setName(layerName)

    // set bottom list and top list
    setConnections(layerParameter, bottoms, nextSize)

    // copy weight and bias
    val (weightBuilder, biasBuilder) = copyParam(module)

    setBlobs(layerParameter, weightBuilder, biasBuilder)

    layerParameter
  }

  private def setConnections(layerParameter : LayerParameter.Builder,
                             bottoms : ArrayBuffer[String], nextSize: Int) : Unit = {

    val layerName = layerParameter.getName

    // set bottom list
    var i = 0
    bottoms.foreach(bottom => {
      // layerParameter.setBottom(i, bottom)
      layerParameter.addBottom(bottom)
      i += 1
    })

    // set top list
    i = 0
    while (i < nextSize) {
      // layerParameter.setTop(i, s"$layerName$i")
      layerParameter.addTop(s"$layerName$i")
      i += 1
    }
  }

  private def setBlobs(layerParameterBuilder: LayerParameter.Builder,
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
        val weightShape = BlobShape.newBuilder
        weight.size().foreach(dim => weightShape.addDim(dim.toLong))
        weightBlobBuilder.setShape(weightShape.build)
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
        val biasShape = BlobShape.newBuilder
        bias.size().foreach(dim => biasShape.addDim(dim.toLong))
        biasBlobBuilder.setShape(biasShape.build)
      }
    }
    (weightBlobBuilder, biasBlobBuilder)
  }

  override protected def getLayerName(layer : GeneratedMessage) : String = {
    layer.asInstanceOf[LayerParameter].getName
  }

  override protected def getLayerType(layer : GeneratedMessage) : String = {
    layer.asInstanceOf[LayerParameter].getType
  }

  protected def getConvolutionParam(layer : GeneratedMessage): Option[ConvolutionParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getConvolutionParam)
  }

  override protected def getLRNParam(layer: GeneratedMessage): Option[LRNParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getLrnParam)
  }

  override protected def getPoolingParam(layer : GeneratedMessage): Option[PoolingParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getPoolingParam)
  }

  protected def getInnerProductParam(layer : GeneratedMessage): Option[InnerProductParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getInnerProductParam)
  }

  protected def getDropoutParam(layer : GeneratedMessage): Option[DropoutParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getDropoutParam)
  }

  protected def getConcatParam(layer : GeneratedMessage): Option[ConcatParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getConcatParam)
  }

  override protected def getPowerParam(layer : GeneratedMessage) : Option[PowerParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getPowerParam)
  }

  override protected def getThresholdParam(layer : GeneratedMessage): Option[ThresholdParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getThresholdParam)
  }

  override protected def getSliceParam(layer : GeneratedMessage): Option[SliceParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getSliceParam)
  }

  override protected def getEltWiseParam(layer : GeneratedMessage): Option[EltwiseParameter] = {
    Some(layer.asInstanceOf[LayerParameter].getEltwiseParam)
  }

  protected def getBlob(layer : GeneratedMessage, ind: Int): Option[Caffe.BlobProto] = {
    if (layer.asInstanceOf[LayerParameter].getBlobsCount > ind) {
      Some(layer.asInstanceOf[LayerParameter].getBlobs(ind))
    } else {
      None
    }
  }
}
