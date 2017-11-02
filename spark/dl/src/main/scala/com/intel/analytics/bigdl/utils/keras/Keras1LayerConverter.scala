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

package com.intel.analytics.bigdl.utils.keras

import com.intel.analytics.bigdl.nn.{Dropout, Linear, Reshape, SpatialConvolution}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import play.api.libs.json.JsNull

import scala.reflect.ClassTag

class Keras1LayerConverter[T: ClassTag](kerasJson: KModel)(implicit ev: TensorNumeric[T])
  extends LayerConverter[T](kerasJson) {


  override def createInput(layer: Layer): AbstractModule[Activity, Activity, T] = {
    // place holder. it should be removed shortly
    return null
  }

  override def createEmbedding(layer: Layer): AbstractModule[Activity, Activity, T] = {
    //    val layerConfig = new EmbeddingConfig(layer)
    //    val lookupTable = LookupTable[T](
    //      nIndex: Int,
    //      nOutput: Int,
    //      paddingValue: Double = 0,
    //    maxNorm: Double = Double.MaxValue,
    //    normType: Double = 2.0,
    //    shouldScaleGradByFreq: Boolean = false,
    //    wRegularizer = RegularizerHelper.toBigDL(layerConfig.wRegularizer)
    //    )
    return null
  }



  override def createConvolution2D(layer: Layer): AbstractModule[Activity, Activity, T] = {
    val layerConfig = new Convolution2DConfig(layer.config)

    val bformat = OrderingHelper.toBigDL(layerConfig.dimOrder)
    val stack_size = bformat match {
      case DataFormat.NCHW => layerConfig.batchInputShape.get(1).toInt
      case DataFormat.NHWC => layerConfig.batchInputShape.get(3).toInt
    }

    val (bpadW, bpadH) = LayerConverter.getPadding(layerConfig.borderMode)

    val blayer = SpatialConvolution[T](
      nInputPlane = stack_size,
      nOutputPlane = layerConfig.nbFilter,
      kernelW = layerConfig.nbCol,
      kernelH = layerConfig.nbRow,
      strideW = layerConfig.subsample.value(0).as[Int],
      // TODO: the first one is col? or they always equal?,
      strideH = layerConfig.subsample.value(1).as[Int],
      padW = bpadW,
      padH = bpadH,
      nGroup = 1,
      propagateBack = true,
      wRegularizer = RegularizerHelper.toBigDL(layerConfig.wRegularizer),
      bRegularizer = RegularizerHelper.toBigDL(layerConfig.bRegularizer.getOrElse(JsNull)),
      withBias = layerConfig.bias.get,
      format = bformat
    )
    ParameterLayerHelper.combo(blayer, layerConfig)
  }

  override def createFlatten(layer: Layer): AbstractModule[Activity, Activity, T] = {
    val layerConfig = new BaseLayerConfig(layer.config)
    val blayer = Reshape[T](
      layerConfig.batchInputShape.get.takeRight(1).map(i => i.toInt))
      .setName(layer.name)
    blayer
  }

  override def createMerge(layer: Layer): AbstractModule[Activity, Activity, T] = {
    return null
  }

  override def createDense(layer: Layer): AbstractModule[Activity, Activity, T] = {
    val layerConfig = new DenseConfig(layer.config)

    val blayer = Linear[T](
      inputSize = layerConfig.inputDim,
      outputSize = layerConfig.outputDim,
      withBias = layerConfig.bias.get,
      wRegularizer = RegularizerHelper.toBigDL(layerConfig.wRegularizer),
      bRegularizer = RegularizerHelper.toBigDL(layerConfig.wRegularizer)
    )
    ParameterLayerHelper.combo(blayer, layerConfig)
  }

  def createDropout(layer: Layer): AbstractModule[Activity, Activity, T] = {
    val dropoutConfig = new DropoutConfig(layer.config)
    Dropout[T](dropoutConfig.p).setName(layer.name)
  }

  def createActivation(layer: Layer): AbstractModule[Activity, Activity, T] = {
    val inboundNodes = layer.inboundNodes
    val instanceName = layer.name
    val layerConfig = new ActivationConfig(layer.config)
    ActivationHelper.toBigDL(layerConfig.activation, layer.name)
  }

}

