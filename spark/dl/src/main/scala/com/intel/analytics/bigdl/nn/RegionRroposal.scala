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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.Convolution2D
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import com.sun.tracing.dtrace.ModuleName
import org.apache.spark.api.java.function

class RegionRroposal(in_channels: Int,
                     anchor_sizes: Array[Float],
                     aspect_ratios: Array[Float],
                     anchor_stride: Array[Float],
                     preNmsTopNTest: Int,
                     postNmsTopNTest: Int,
                     nms_thread: Float,
                     min_size: Int,
                     rpnPostNmsTopNTrain: Int) extends AbstractModule[Table, Tensor[Float], Float] {

  val anchor_generator = new AnchorGenerate(anchor_sizes, aspect_ratios, anchor_stride)
  val numAnchors = anchor_generator.num_anchors_per_location()
  val head = new RPNHead(in_channels, numAnchors)
  val box_selector_test = new RPNPostProcessor(preNmsTopNTest, postNmsTopNTest,
    nms_thread, min_size, rpnPostNmsTopNTrain)


  /**
   * input is a table and contains:
   * first tensor: images: images for which we want to compute the predictions
   * second tensor: features: features computed from the images that are used for
   *  computing the predictions.
   */
  override def updateOutput(input: Table): Tensor[Float] = {
    require(input.length() == 2 && !this.isTraining(), "Only support tests")
    val images = input[Tensor[Float]](1)
    val features = input[Tensor[Float]](2)

    val anchorsOutput = anchor_generator.forward(T(features, images))
    val anchors = anchorsOutput[Table](1)
    val headOutput = head.forward(features)
    val objectness = headOutput.apply[Tensor[Float]](1)
    val rpn_box_regression = headOutput.apply[Tensor[Float]](2)

    output = box_selector_test.forward(T(anchors[Tensor[Float]](1), objectness,
      rpn_box_regression, anchors[Tensor[Float]](2)))

    println("done")
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    gradInput = null
    gradInput
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    head.parameters()
  }
}
