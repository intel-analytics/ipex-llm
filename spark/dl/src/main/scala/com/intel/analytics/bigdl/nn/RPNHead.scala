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
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

// Adds a simple RPN Head with classification and regression heads
class RPNHead[T: ClassTag](in_channels: Int, num_anchors: Int)(implicit ev: TensorNumeric[T])
  extends BaseModule {

  override def buildModel(): Module[T] = {
    val conv = SpatialConvolution[T](in_channels, in_channels,
      kernelH = 3, kernelW = 3, strideH = 1, strideW = 1, padH = 1, padW = 1)
    conv.weight.apply1(_ => ev.fromType(RNG.normal(0, 0.01)))
    conv.bias.apply1(_ => ev.zero)
    val relu = ReLU[T]()
    val conv2 = SpatialConvolution[T](in_channels, num_anchors,
      kernelH = 1, kernelW = 1, strideH = 1, strideW = 1).setName(this.getName() + "_cls_logits")
    conv2.weight.apply1(_ => ev.fromType(RNG.normal(0, 0.01)))
    conv2.bias.apply1(_ => ev.zero)
    val conv3 = SpatialConvolution[T](in_channels, num_anchors * 4,
      kernelH = 1, kernelW = 1, strideH = 1, strideW = 1).setName(this.getName() + "_bbox_pred")
    conv3.weight.apply1(_ => ev.fromType(RNG.normal(0, 0.01)))
    conv3.bias.apply1(_ => ev.zero)

    val input = Input()
    val node1 = conv.inputs(input)
    val node2 = relu.inputs(node1)
    val node3 = conv2.inputs(node2)
    val node4 = conv3.inputs(node2)

    Graph(input, Array(node3, node4))
  }
}
