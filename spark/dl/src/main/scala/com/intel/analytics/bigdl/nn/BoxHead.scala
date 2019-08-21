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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}

class BoxHead(
  val inChannels: Int = 0,
  val resolution: Int = 0,
  val scales: Array[Float],
  val samplingRratio: Float = 2.0f,
  val scoreThresh: Float = 0.05f,
  val nmsThresh: Float = 0.5f,
  val detections_per_img: Int = 100,
  val representation_size: Int = 1024,
  val numClasses: Int = 81 // coco dataset class number
  )(implicit ev: TensorNumeric[Float])
  extends BaseModule[Float] {

  val featureExtractor = new FPN2MLPFeatureExtractor(
    inChannels, resolution, scales, samplingRratio.toInt, representation_size)

  val predictor = new FPNPredictor(numClasses, representation_size)
  val postProcessor = new FPNPostProcessor(scoreThresh, nmsThresh, detections_per_img)

    override def buildModel(): Module[Float] = {
      val featureExtractor = new FPN2MLPFeatureExtractor(
        inChannels, resolution, scales, samplingRratio.toInt, representation_size)

      val predictor = new FPNPredictor(numClasses, representation_size)
      val postProcessor = new FPNPostProcessor(scoreThresh, nmsThresh, detections_per_img)

      val features = Input()
      val proposals = Input()

      val maskFeatures = featureExtractor.inputs(features, proposals)
      val maskLogits = predictor.inputs(maskFeatures)
      val result = postProcessor.inputs(maskLogits, proposals)

      Graph(Array(features, proposals), Array(maskFeatures, result))
    }

  private[nn] def clsPredictor(numClass: Int,
                               inChannels: Int): Module[Float] = {
    val cls_score = Linear[Float](inChannels, numClass)
    // todo: check with torch
    cls_score.weight.apply1(_ => RNG.normal(0, 0.01).toFloat)
    cls_score.bias.fill(0.0f)
    cls_score.asInstanceOf[Module[Float]]
  }

  private[nn] def bboxPredictor(numClass: Int,
                               inChannels: Int): Module[Float] = {
    val bbox_pred = Linear[Float](inChannels, numClass * 4)
    bbox_pred.weight.apply1(_ => RNG.normal(0, 0.001).toFloat)
    bbox_pred.bias.fill(0.0f)
    bbox_pred.asInstanceOf[Module[Float]]
  }

  private[nn] def maskFeatureExtractor(inChannels: Int,
                                       resolution: Int,
                                       scales: Array[Float],
                                       samplingRatio: Float,
                                       layers: Array[Int],
                                       dilation: Int,
                                       useGn: Boolean = false): Module[Float] = {

    require(dilation == 1, s"Only support dilation = 1, but get ${dilation}")

    val model = Sequential[Float]()
    model.add(Pooler(resolution, scales, samplingRatio.toInt))

    var nextFeatures = inChannels
    var i = 0
    while (i < layers.length) {
      val features = layers(i)
      // todo: not support dilation convolution
      val module = SpatialConvolution[Float](
        nextFeatures,
        features,
        kernelW = 3,
        kernelH = 3,
        strideW = 1,
        strideH = 1,
        padW = dilation,
        padH = dilation,
        withBias = if (useGn) false else true
      ).setName(s"mask_fcn{${i}}")

      // weight init
      module.setInitMethod(MsraFiller(false), Zeros)
      model.add(module)
      nextFeatures = features
      i += 1
    }
    model.add(ReLU[Float]())
  }

  override def updateOutput(input: Activity): Activity = {
    val features = input.toTable[Table](1)
    val proposals = input.toTable[Tensor[Float]](2)

    val x = featureExtractor.forward(T(features, proposals))
    val y = predictor.forward(x)
    val z = postProcessor.forward(T(y, proposals))

//    output = model.updateOutput(input)
    output
  }
}
