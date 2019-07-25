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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}

// Adds a simple RPN Head with classification and regression heads
class RPNHead(in_channels: Int, num_anchors: Int)
  extends AbstractModule[Tensor[Float], Table, Float]{

  // todo: check Convolution2D
  private val conv = SpatialConvolution[Float](in_channels, in_channels,
    kernelH = 3, kernelW = 3, strideH = 1, strideW = 1, padH = 1, padW = 1)
  conv.weight.apply1(_ => RNG.normal(0, 0.01).toFloat)
  conv.bias.apply1(_ => 0.0f)
  private val relu = ReLU[Float]()

  private val cls_logits = SpatialConvolution[Float](in_channels, num_anchors,
    kernelH = 1, kernelW = 1, strideH = 1, strideW = 1)
  cls_logits.weight.apply1(_ => RNG.normal(0, 0.01).toFloat)
  cls_logits.bias.apply1(_ => 0.0f)

  private val bbox_pred = SpatialConvolution[Float](in_channels, num_anchors * 4,
    kernelH = 3, kernelW = 3, strideH = 1, strideW = 1)
  bbox_pred.weight.apply1(_ => RNG.normal(0, 0.01).toFloat)
  bbox_pred.bias.apply1(_ => 0.0f)

  override def updateOutput(input: Tensor[Float]): Table = {
    val conv_res = conv.forward(input)
    val relu_res = relu.forward(conv_res)
    val logits_res = cls_logits.forward(relu_res)
    val bbox_res = bbox_pred.forward(relu_res)
    output = T(logits_res, bbox_res)
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Table): Tensor[Float] = {
    gradInput
  }
}
