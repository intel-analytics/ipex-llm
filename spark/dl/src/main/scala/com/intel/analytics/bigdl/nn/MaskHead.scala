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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

class MaskHead(
  val inChannels: Int,
  val resolution: Int,
  val scales: Array[Float],
  val samplingRatio: Int,
  val layers: Array[Int],
  val dilation: Int,
  val numClasses: Int,
  val useGn: Boolean = false)(implicit ev: TensorNumeric[Float])
  extends BaseModule[Float] {

  override def buildModel(): Module[Float] = {
    val featureExtractor = this.maskFeatureExtractor(
      inChannels, resolution, scales, samplingRatio, layers, dilation, useGn)
    val dimReduced = layers(layers.length - 1)
    val predictor = this.maskPredictor(dimReduced, numClasses, dimReduced)
    val postProcessor = new MaskPostProcessor()

    /**
     * input: feature-maps from possibly several levels and proposal boxes
     * return:
     * first tensor: the result of the feature extractor
     * second tensor: proposals (list[BoxList]): during training, the original proposals
     *      are returned. During testing, the predicted boxlists are returned
     *      with the `mask` field set
     */
    val features = Input()
    val proposals = Input()
    val labels = Input()

    val maskFeatures = featureExtractor.inputs(features, proposals)
    val maskLogits = predictor.inputs(maskFeatures)
    val result = postProcessor.inputs(maskLogits, labels)

    Graph(Array(features, proposals, labels), Array(maskFeatures, result))
  }

  private[nn] def maskPredictor(inChannels: Int,
                                numClasses: Int,
                                dimReduced: Int): Module[Float] = {
    val convMask = SpatialFullConvolution(inChannels, dimReduced,
      kW = 2, kH = 2, dW = 2, dH = 2)
    val maskLogits = SpatialConvolution(nInputPlane = dimReduced,
      nOutputPlane = numClasses, kernelW = 1, kernelH = 1, strideH = 1, strideW = 1)

    // init weight & bias, MSRAFill by default
    convMask.setInitMethod(MsraFiller(false), Zeros)
    maskLogits.setInitMethod(MsraFiller(false), Zeros)

    val model = Sequential[Float]()
    model.add(convMask).add(ReLU[Float]()).add(maskLogits)
    model
  }

  private[nn] def maskFeatureExtractor(inChannels: Int,
                                       resolution: Int,
                                       scales: Array[Float],
                                       samplingRatio: Int,
                                       layers: Array[Int],
                                       dilation: Int,
                                       useGn: Boolean = false): Module[Float] = {

    require(dilation == 1, s"Only support dilation = 1, but got ${dilation}")

    val model = Sequential[Float]()
    model.add(Pooler(resolution, scales, samplingRatio))

    var nextFeatures = inChannels
    var i = 0
    while (i < layers.length) {
      val features = layers(i)
      // todo: support dilation convolution with no bias
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
      ).setName(s"mask_fcn${i + 1}")

      // weight init
      module.setInitMethod(MsraFiller(false), Zeros)
      model.add(module).add(ReLU[Float]())
      nextFeatures = features
      i += 1
    }
    model
  }
}

private[nn] class MaskPostProcessor()(implicit ev: TensorNumeric[Float])
  extends AbstractModule[Table, Tensor[Float], Float] {

  @transient var rangeBuffer: Tensor[Float] = null
  private val sigmoid = Sigmoid[Float]()

  /**
   * @param input feature-maps from possibly several levels, proposal boxes and labels
   * @return the predicted boxlists are returned with the `mask` field set
   */
  override def updateOutput(input: Table): Tensor[Float] = {
    val maskLogits = input[Tensor[Float]](1)
    val labels = input[Tensor[Float]](2)

    val num_masks = maskLogits.size(1)
    if (rangeBuffer == null || rangeBuffer.nElement() != num_masks) {
      rangeBuffer = Tensor[Float](num_masks)
      rangeBuffer.range(0, num_masks - 1, 1)
    }

    val mask_prob = sigmoid.forward(maskLogits)
    require(labels.nDimension() == 1, s"Labels should be tensor with one dimension," +
      s"but get ${labels.nDimension()}")
    require(rangeBuffer.nElement() == labels.nElement(), s"number of masks should be same" +
      s"with labels, but get ${rangeBuffer.nElement()} ${labels.nElement()}")

    output.resize(rangeBuffer.nElement(), 1, mask_prob.size(3), mask_prob.size(4))

    var i = 1
    while (i <= rangeBuffer.nElement()) {
      val dim = rangeBuffer.valueAt(i).toInt + 1
      val index = labels.valueAt(i).toInt // start from 1
      output.narrow(1, i, 1).copy(mask_prob.narrow(1, i, 1).narrow(2, index + 1, 1))
      i += 1
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    throw new UnsupportedOperationException("MaskPostProcessor only support inference")
  }
}

object MaskHead {
  def apply(inChannels: Int,
  resolution: Int = 14,
  scales: Array[Float] = Array[Float](0.25f, 0.125f, 0.0625f, 0.03125f),
  samplingRratio: Int = 2,
  layers: Array[Int] = Array[Int](256, 256, 256, 256),
  dilation: Int = 1,
  numClasses: Int = 81,
  useGn: Boolean = false)(implicit ev: TensorNumeric[Float]): Module[Float] = {
    new MaskHead(inChannels, resolution, scales, samplingRratio,
      layers, dilation, numClasses, useGn)
  }
}

