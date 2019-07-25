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

import breeze.linalg.*
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}

class AnchorGenerate(
  anchor_sizes: Array[Float],
  aspect_ratios: Array[Float],
  anchor_stride: Array[Float],
  straddle_thresh: Int = 0) extends AbstractModule[Table, Table, Float]{

  require(anchor_sizes.length == anchor_stride.length)
  val scalesForStride = Array[Float](8f)
  val cls = Anchor(aspect_ratios, scalesForStride)

/**
 * @param input input(1): features map
 * input(2): image list
 * return: BBox information, Table with first tensor is proposals, and nexts is image info
 */
 override def updateOutput(input: Table): Table = {
    val featuresMap = input[Tensor[Float]](1)
    val imageList = input[Tensor[Float]](2)

   require(featuresMap.size(1) == anchor_sizes.length)
   require(featuresMap.size(1) == imageList.size(1))

   val zipArr = anchor_sizes.zip(anchor_stride)
   var i = 0
   while (i < zipArr.length) {
     val size = anchor_sizes(i)
     val stride = anchor_stride(i)
     val feature = featuresMap.select(1, i + 1).size()
     val res = cls.generateAnchors(feature(0), feature(1), stride).clone()
     output(i + 1) = T(res, imageList.select(1, i + 1))
     i += 1
   }
   output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = null
    gradInput
  }
}
