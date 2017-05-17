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
import caffe.Caffe.V1LayerParameter.LayerType
import caffe.Caffe._
import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.{BatchNormalization, Linear, SpatialConvolution, View}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


class V1LayerConverter[T: ClassTag](implicit ev: TensorNumeric[T]) extends Converter[T] {

  override protected def fromCaffeConvolution(layer : GeneratedMessage) : ModuleNode[T] = {
    val param = getConvolutionParam(layer).get
    val weightBlob = getBlob(layer, 0).get
    val group = if (param.getGroup == 0)  1 else param.getGroup
    val nInputPlane = weightBlob.getChannels * group
    val nOutPlane = weightBlob.getNum
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
    new SpatialConvolution[T](nInputPlane, nOutPlane, kw, kh, dw, dh, pw, ph, group)
      .setName(getLayerName(layer)).apply()
  }

  override protected def fromCaffeInnerProduct(layer : GeneratedMessage) : ModuleNode[T] = {
    val param = getInnerProductParam(layer).get
    val withBias = param.getBiasTerm
    val layerName = getLayerName(layer)
    val weightBlob = getBlob(layer.asInstanceOf[V1LayerParameter], 0).get
    var nInputPlane = weightBlob.getWidth
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

  // No implementation in V1
  override protected def fromCaffeBatchNormalization(layer : GeneratedMessage) : ModuleNode[T] = {
    null
  }

  // No implementation in V1
  override protected def fromCaffeELU(layer : GeneratedMessage) : ModuleNode[T] = {
    null
  }

  // No implementation in V1
  override protected def fromCaffeReshape(layer : GeneratedMessage) : ModuleNode[T] = {
    null
  }

  // No implementation in V1
  override protected def fromCaffeScale(layer : GeneratedMessage) : ModuleNode[T] = {
    null
  }

  // No implementation in V1
  override protected def fromCaffeBias(layer : GeneratedMessage) : ModuleNode[T] = {
    null
  }

  // No implementation in V1
  override protected def fromCaffeTile(layer : GeneratedMessage) : ModuleNode[T] = {
    null
  }

  override protected def toCaffeConvolution(module : ModuleNode[T],
                                            bottoms : ArrayBuffer[String]): GeneratedMessage = {
    val layerParameter = V1LayerParameter.newBuilder()

    val layerName = module.element.getName

    layerParameter.setName(layerName)

    layerParameter.setType(LayerType.CONVOLUTION)

    // set bottom list and top list
    setConnections(layerParameter, bottoms, module.nextNodes)

    // copy weight and bias
    var (weightBuilder, biasBuilder) = copyParam(module.element, layerParameter)

    // get convolution param map
    val layerParams = toCaffeConvolutionParam(module.element)

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

    val layerParameter = V1LayerParameter.newBuilder()

    val layerName = moduleNode.element.getName

    layerParameter.setName(layerName)

    layerParameter.setType(LayerType.RELU)

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
    val layerParameter = V1LayerParameter.newBuilder()

    val layerName = moduleNode.element.getName

    layerParameter.setName(layerName)

    layerParameter.setType(LayerType.LRN)

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
    val layerParameter = V1LayerParameter.newBuilder()

    val layerName = moduleNode.element.getName

    layerParameter.setName(layerName)

    layerParameter.setType(LayerType.POOLING)

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
    val layerParameter = V1LayerParameter.newBuilder()

    val layerName = moduleNode.element.getName

    layerParameter.setName(layerName)

    layerParameter.setType(LayerType.INNER_PRODUCT)

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
    val layerParameter = toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).
      setType(LayerType.DROPOUT)

    val dropOutRatio = toCaffeDropOutParam(moduleNode.element)

    val dropoutParameter = DropoutParameter.newBuilder

    dropoutParameter.setDropoutRatio(dropOutRatio.toFloat)

    layerParameter.setDropoutParam(dropoutParameter.build)

    layerParameter.build

  }

  override protected def toCaffeLogSoftMax(moduleNode : ModuleNode[T],
                                           bottoms : ArrayBuffer[String]): GeneratedMessage = {

    toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType(LayerType.SOFTMAX).build
  }

  override protected def toCaffeTanh(moduleNode : ModuleNode[T],
                                     bottoms : ArrayBuffer[String]): GeneratedMessage = {

    toCaffeWithWeightAndBiasOnly(moduleNode, bottoms).setType(LayerType.TANH).build
  }

  private def toCaffeWithWeightAndBiasOnly(moduleNode : ModuleNode[T],
    bottoms : ArrayBuffer[String]): V1LayerParameter.Builder = {

    val layerParameterBuilder = V1LayerParameter.newBuilder()

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

  private def setConnections(layerParameter : V1LayerParameter.Builder,
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
    builder : V1LayerParameter.Builder) :
    (BlobProto.Builder, BlobProto.Builder) = {
    val name = module.getName
    val params = module.getParametersTable()
    val weight = params[Tensor[T]]("weight")
    val weightData = weight.storage().array()
    var i = 0
    val weightBlobBuilder = BlobProto.newBuilder()
    while (i < weightData.length) {
      weightBlobBuilder.setData(i, ev.toType(weightData(i)))
      i += 1
    }
    val bias = params[Tensor[T]]("bias")
    val biasData = bias.storage().array()
    i = 0
    val biasBlobBuilder = BlobProto.newBuilder()
    while (i < biasData.length) {
      biasBlobBuilder.setData(i, ev.toType(biasData(i)))
      i += 1
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

  private def getBlob(layer : GeneratedMessage, ind: Int): Option[Caffe.BlobProto] = {
    if (layer.asInstanceOf[V1LayerParameter].getBlobsCount > ind) {
      Some(layer.asInstanceOf[V1LayerParameter].getBlobs(ind))
    } else {
      None
    }
  }
}
