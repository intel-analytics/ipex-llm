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

import java.util

import caffe.Caffe
import caffe.Caffe.EltwiseParameter.EltwiseOp
import caffe.Caffe.{PoolingParameter, _}
import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


class LayerConverter[T: ClassTag](implicit ev: TensorNumeric[T]) extends Converter[T]{

  override protected def fromCaffeConvolution(layer : GeneratedMessage) : ModuleNode[T] = {
    val name = getLayerName(layer)
    val param = getConvolutionParam(layer).get
    val group = if (param.getGroup == 0)  1 else param.getGroup
    val  weightBlob = getBlob(layer, 0).get
    val nInputPlane = if (weightBlob.hasShape) weightBlob.getShape.getDim(1)
      else weightBlob.getChannels * group
    val nOutPlane = if (weightBlob.hasShape) weightBlob.getShape.getDim(0)
      else weightBlob.getNum
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
    new SpatialConvolution[T](nInputPlane.toInt, nOutPlane.toInt, kw, kh, dw, dh, pw, ph, group)
      .setName(getLayerName(layer)).apply()
  }

  override protected def fromCaffeInnerProduct(layer : GeneratedMessage) : ModuleNode[T] = {
    val param = getInnerProductParam(layer).get
    val withBias = param.getBiasTerm
    val layerName = getLayerName(layer)
    val weightBlob = getBlob(layer.asInstanceOf[LayerParameter], 0).get
    var nInputPlane = 0
    if (weightBlob.hasShape) {
      nInputPlane = weightBlob.getShape.getDim(1).toInt
    }
    else {
      nInputPlane = weightBlob.getWidth
    }
    val nOutputPlane = param.getNumOutput
    val linear = Linear[T](nInputPlane, nOutputPlane, withBias = withBias).setName(layerName)
    val node = linear.apply()
    if(nInputPlane != nOutputPlane) {
      // Construct a view layer in between
      val view = View[T](nInputPlane).apply()
      view -> node
      view
    } else {
      node
    }
  }

  override protected def fromCaffeBatchNormalization(layer : GeneratedMessage) : ModuleNode[T] = {
    val param = layer.asInstanceOf[LayerParameter].getBatchNormParam
    val eps = param.getEps
    BatchNormalization[T](3, eps).apply()
  }

  override protected def fromCaffeELU(layer : GeneratedMessage) : ModuleNode[T] = {
    val param = layer.asInstanceOf[LayerParameter].getEluParam
    var alpha = 1.0
    if (param.hasAlpha) alpha = param.getAlpha
    ELU[T](alpha).apply()
  }

  override protected def fromCaffeReshape(layer : GeneratedMessage) : ModuleNode[T] = {
    val param = layer.asInstanceOf[LayerParameter].getReshapeParam
    val shapeSize = param.getShape.getDimList.toArray.asInstanceOf[Array[Int]]
    Reshape[T](shapeSize).setName(getLayerName(layer)).apply()
  }

  override protected def fromCaffeScale(layer : GeneratedMessage) : ModuleNode[T] = {
    val param = layer.asInstanceOf[LayerParameter].getScaleParam
    val layerName = getLayerName(layer)
    // second blob as weight for scale
    val weightBlob = getBlob(layer, 1)
    if (weightBlob.isDefined) {
      val blob = weightBlob.get
      val size = blob.getShape.getDimList.toArray.asInstanceOf[Array[Int]]
      Scale[T](size).setName(layerName).apply()
    } else {
      val inputBlob = getBlob(layer, 0).get
      val shape = inputBlob.getShape
      val axis = param.getAxis
      var numOfAxis = param.getNumAxes
      if (numOfAxis == -1) {
        numOfAxis = shape.getDimList.size() - 1
      } else {
        numOfAxis = numOfAxis + axis
      }
      val size = shape.getDimList.subList(axis, numOfAxis).asInstanceOf[Array[Int]]
      Scale[T](size).setName(layerName).apply()
    }
  }

  override protected def fromCaffeBias(layer : GeneratedMessage) : ModuleNode[T] = {
    val param = layer.asInstanceOf[LayerParameter].getBiasParam
    // input blob
    val weightBlob = getBlob(layer, 0)
    val size = weightBlob.get.getShape.getDimList.toArray().asInstanceOf[Array[Int]].product
    Add[T](size).setName(getLayerName(layer)).apply()
  }

  override protected def fromCaffeTile(layer : GeneratedMessage) : ModuleNode[T] = {
    val param = layer.asInstanceOf[LayerParameter].getTileParam
    val axis = param.getAxis
    val tiles = param.getTiles
    Replicate[T](tiles, axis).setName(getLayerName(layer)).apply()
  }

  override protected def toCaffeConvolution(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = LayerParameter.newBuilder()

    val layerName = moduleNode.element.getName

    layerParameter.setName(layerName)

    layerParameter.setType("Convolution")

    // set bottom list and top list
    setConnections(layerParameter, bottoms, moduleNode.nextNodes)

    // copy weight and bias
    var (weightBuilder, biasBuilder) = copyParam(moduleNode.element, layerParameter)

    // get convolution param map
    val layerParams = toCaffeConvolutionParam(moduleNode.element)

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
    weightBuilder.setChannels(nInputPlane / ngroup)
    weightBuilder.setNum(nOutputPlane)

    layerParameter.setBlobs(0, weightBuilder.build)
    layerParameter.setBlobs(1, biasBuilder.build)
    layerParameter.setConvolutionParam(convolutionParam.build)

    // build concolution layer
    layerParameter.build()

  }

  override protected def toCaffeRelu(moduleNode : ModuleNode[T],
                            bottoms : ArrayBuffer[String]): GeneratedMessage = {

    val layerParameter = LayerParameter.newBuilder()

    val layerName = moduleNode.element.getName

    layerParameter.setName(layerName)

    layerParameter.setType("ReLU")

    // set bottom list and top list
    setConnections(layerParameter, bottoms, moduleNode.nextNodes)

    // copy weight and bias
    var (weightBuilder, biasBuilder) = copyParam(moduleNode.element, layerParameter)
    layerParameter.setBlobs(0, weightBuilder.build)
    layerParameter.setBlobs(1, biasBuilder.build)

    // build concolution layer
    layerParameter.build()
  }

  override protected def toCaffeLRN(moduleNode : ModuleNode[T],
                                    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = LayerParameter.newBuilder()

    val layerName = moduleNode.element.getName

    layerParameter.setName(layerName)

    layerParameter.setType("LRN")

    // set bottom list and top list
    setConnections(layerParameter, bottoms, moduleNode.nextNodes)

    // copy weight and bias
    val (weightBuilder, biasBuilder) = copyParam(moduleNode.element, layerParameter)

    val (localSize, alpha, belta, k) = toCaffeLRNParam(moduleNode.element)

    val lrnParameter = LRNParameter.newBuilder()

    lrnParameter.setLocalSize(localSize)
    lrnParameter.setAlpha(alpha.toFloat)
    lrnParameter.setBeta(belta.toFloat)
    lrnParameter.setK(k.toFloat)

    layerParameter.setBlobs(0, weightBuilder.build)
    layerParameter.setBlobs(1, biasBuilder.build)

    layerParameter.setLrnParam(lrnParameter.build)

    layerParameter.build
  }

  override protected def toCaffeMaxPooling(moduleNode : ModuleNode[T],
                                           bottoms : ArrayBuffer[String]): GeneratedMessage = {
    toCaffePooling(moduleNode, bottoms, true)
  }

  override protected def toCaffeAvePooling(moduleNode : ModuleNode[T],
                                           bottoms : ArrayBuffer[String]): GeneratedMessage = {
    toCaffePooling(moduleNode, bottoms, false)
  }

  private def toCaffePooling(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String], max : Boolean): GeneratedMessage = {
    val layerParameter = LayerParameter.newBuilder()

    val layerName = moduleNode.element.getName

    layerParameter.setName(layerName)

    layerParameter.setType("Pooling")

    // set bottom list and top list
    setConnections(layerParameter, bottoms, moduleNode.nextNodes)

    // copy weight and bias
    val (weightBuilder, biasBuilder) = copyParam(moduleNode.element, layerParameter)

    val poolingParameter = if (max) toCaffeMaxPoolingParam(moduleNode.element)
      else toCaffeAvgPoolingParam(moduleNode.element)

    layerParameter.setBlobs(0, weightBuilder.build)
    layerParameter.setBlobs(1, biasBuilder.build)

    layerParameter.setPoolingParam(poolingParameter)

    layerParameter.build
  }

  override protected def toCaffeInnerProduct(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = LayerParameter.newBuilder()

    val layerName = moduleNode.element.getName

    layerParameter.setName(layerName)

    layerParameter.setType("InnerProduct")

    // set bottom list and top list
    setConnections(layerParameter, bottoms, moduleNode.nextNodes)

    // copy weight and bias
    val (weightBuilder, biasBuilder) = copyParam(moduleNode.element, layerParameter)

    val (inputSize, outputSize, withBias) = toCaffeInnerProductParam(moduleNode.element)

    weightBuilder.setWidth(inputSize)

    val innerProductParameter = InnerProductParameter.newBuilder

    innerProductParameter.setNumOutput(outputSize)

    innerProductParameter.setBiasTerm(withBias)

    layerParameter.setBlobs(0, weightBuilder.build)
    layerParameter.setBlobs(1, biasBuilder.build)

    layerParameter.setInnerProductParam(innerProductParameter.build)

    layerParameter.build
  }

  override protected def toCaffeDropOut(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {

    val layerParameter = toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Dropout")

    val dropOutRatio = toCaffeDropOutParam(moduleNode.element)

    val dropoutParameter = DropoutParameter.newBuilder

    dropoutParameter.setDropoutRatio(dropOutRatio.toFloat)

    layerParameter.setDropoutParam(dropoutParameter.build)

    layerParameter.build
  }

  override protected def toCaffeLogSoftMax(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {

    toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Softmax").build
  }

  override protected def toCaffeTanh(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {

    toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("TANH").build
  }

  override protected def toCaffeSigmoid(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Sigmoid").build
  }

  override protected def toCaffeAbs(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Abs").build
  }

  override protected def toCaffeBatchNormalization(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("BatchNorm")
    val batchNormParameter = BatchNormParameter.newBuilder()
    val eps = toCaffeBatchNormParam(moduleNode.element)
    batchNormParameter.setEps(eps.toFloat)
    layerParameter.setBatchNormParam(batchNormParameter.build)
    layerParameter.build
  }

  override protected def toCaffeConcat(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Concat")
    val dimension = toCaffeConcatParam(moduleNode.element)
    val concatParameter = ConcatParameter.newBuilder
    concatParameter.setAxis(dimension - 1)
    layerParameter.setConcatParam(concatParameter.build)
    layerParameter.build
  }

  override protected def toCaffeElu(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Concat")
    layerParameter.setEluParam(toCaffeEluParam(moduleNode.element))
    layerParameter.build
  }

  override protected def toCaffeFlattern(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Flattern").build
  }

  override protected def toCaffeLog(moduleNode : ModuleNode[T],
                           bottoms : ArrayBuffer[String]): GeneratedMessage = {
    toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Log").build
  }

  override protected def toCaffePower(moduleNode : ModuleNode[T],
                             bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Power")
    layerParameter.setPowerParam(toCaffePowerParam(moduleNode.element))
    layerParameter.build
  }

  override protected def toCaffePReLu(moduleNode : ModuleNode[T],
                             bottoms : ArrayBuffer[String]): GeneratedMessage = {
    toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("PReLU").build
  }

  override protected def toCaffeRecurrent(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Recurrent").build
  }

  override protected def toCaffeReshape(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Reshape")
    layerParameter.setReshapeParam(toCaffeReshapeParam(moduleNode.element))
    layerParameter.build
  }

  override protected def toCaffeScale(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = LayerParameter.newBuilder()

    val layerName = moduleNode.element.getName

    layerParameter.setName(layerName)

    layerParameter.setType("Scale")

    // set bottom list and top list
    setConnections(layerParameter, bottoms, moduleNode.nextNodes)

    // copy weight and bias
    var (weightBuilder, biasBuilder) = copyParam(moduleNode.element, layerParameter)

    val blobShape = toCaffeScalaParam(moduleNode.element)

    biasBuilder.setShape(blobShape)

    layerParameter.setBlobs(0, weightBuilder.build)
    layerParameter.setBlobs(1, biasBuilder.build)

    layerParameter.build
  }

  override protected def toCaffeBias(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Bias").build
  }

  override  protected def toCaffeThreshold(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Threshold")
    val thresholdParameter = toCaffeThresholdParam(moduleNode.element)
    layerParameter.setThresholdParam(thresholdParameter)
    layerParameter.build
  }

  override protected def toCaffeExp(moduleNode : ModuleNode[T],
                                    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Exp").build
  }

  override protected def toCaffeSlice(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Slice")
    val sliceParameter = toCaffeSliceParam(moduleNode.element)
    layerParameter.setSliceParam(sliceParameter)
    layerParameter.build
  }

  override protected def toCaffeTile(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("Tile")
    val tileParameter = toCaffeTileParam(moduleNode.element)
    layerParameter.setTileParam(tileParameter.toBuilder)
    layerParameter.build
  }

  override protected def toCaffeEltWiseMax(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("EltWise")
    val eltwiseParameter = EltwiseParameter.newBuilder
    eltwiseParameter.setOperation(EltwiseOp.MAX)
    layerParameter.setEltwiseParam(eltwiseParameter)
    layerParameter.build
  }

  override protected def toCaffeEltWiseAdd(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("EltWise")
    val eltwiseParameter = EltwiseParameter.newBuilder
    eltwiseParameter.setOperation(EltwiseOp.SUM)
    eltwiseParameter.setCoeff(1, 1.0f)
    layerParameter.setEltwiseParam(eltwiseParameter)
    layerParameter.build
  }

  override protected def toCaffeEltWiseSub(moduleNode : ModuleNode[T],
                                           bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType("EltWise")
    val eltwiseParameter = EltwiseParameter.newBuilder
    eltwiseParameter.setOperation(EltwiseOp.SUM)
    eltwiseParameter.setCoeff(1, -1.0f)
    layerParameter.setEltwiseParam(eltwiseParameter)
    layerParameter.build
  }

  private def toCaffeWithWeightAndBiasOnly(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): LayerParameter.Builder = {

    val layerParameterBuilder = LayerParameter.newBuilder()

    val layerName = moduleNode.element.getName

    layerParameterBuilder.setName(layerName)

    // set bottom list and top list
    setConnections(layerParameterBuilder, bottoms, moduleNode.nextNodes)

    // copy weight and bias
    val (weightBuilder, biasBuilder) = copyParam(moduleNode.element, layerParameterBuilder)

    layerParameterBuilder.setBlobs(0, weightBuilder.build)
    layerParameterBuilder.setBlobs(1, biasBuilder.build)

    layerParameterBuilder
  }

  private def setConnections(layerParameter : LayerParameter.Builder,
    bottoms : ArrayBuffer[String], nextNodes: Seq[ModuleNode[T]]) : Unit = {

    val layerName = layerParameter.getName

    // set bottom list
    var i = 0
    bottoms.foreach(bottom => {
      layerParameter.setBottom(i, bottom)
      i += 1
    })

    // set top list
    i = 0
    while (i < nextNodes.size) {
      layerParameter.setTop(i, s"$layerName$i")
      i += 1
    }
  }

  private def copyParam(module : AbstractModule[Activity, Tensor[T], T],
    builder : LayerParameter.Builder) : (BlobProto.Builder, BlobProto.Builder) = {
    val name = module.getName
    val params = module.getParametersTable()
    val weight = params[Tensor[T]]("weight")
    val weightData = weight.storage().array()
    var i = 0
    val weightBlobBuilder = BlobProto.newBuilder()
    while (i < weightData.length) {
      weightBlobBuilder.setData(i, ev.toType[Float](weightData(i)))
      i += 1
    }
    val bias = params[Tensor[T]]("bias")
    val biasData = bias.storage().array()
    i = 0
    val biasBlobBuilder = BlobProto.newBuilder()
    while (i < biasData.length) {
      biasBlobBuilder.setData(i, ev.toType[Float](biasData(i)))
      i += 1
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

  private def getBlob(layer : GeneratedMessage, ind: Int): Option[Caffe.BlobProto] = {
    if (layer.asInstanceOf[LayerParameter].getBlobsCount > ind) {
      Some(layer.asInstanceOf[LayerParameter].getBlobs(ind))
    } else {
      None
    }
  }
}
