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
import com.intel.analytics.bigdl.utils.Table

class FPN(in_channels_list: Array[Int], out_channels: Int)
  extends AbstractModule[Table, Table, Float]{

  private val num_feature_maps = in_channels_list.length
  private var inner_blocks_modules = new Array[SpatialConvolution[Float]](num_feature_maps)
  private var layer_blocks_modules = new Array[SpatialConvolution[Float]](num_feature_maps)

  def init(): Unit = {
    for (i <- 0 to num_feature_maps - 1) {
      if (in_channels_list(i) != 0) {
        val inner_block_module =
          SpatialConvolution[Float](in_channels_list(i), out_channels, 1, 1, 1, 1)
        val layer_block_module =
          SpatialConvolution[Float](out_channels, out_channels, 3, 3, 1, 1)
        inner_blocks_modules(i) = inner_block_module
        layer_blocks_modules(i) = layer_block_module
      }
    }
  }

  override def updateOutput(input: Table): Table = {
    init()
    var last_inner = inner_blocks_modules(num_feature_maps - 1).forward(input(num_feature_maps - 1))
    var results = new Table()
    results.insert(layer_blocks_modules(num_feature_maps - 1).forward(last_inner))
    for (i <- num_feature_maps - 2 to 0 by -1) {
      val feature = input(i)
      val inner_block = inner_blocks_modules(i)
      val layer_block = layer_blocks_modules(i)
      if (layer_block != null) {
        val inner_topdown = UpSampling2D(Array(2, 2)).forward(last_inner)
        val inner_lateral = inner_block.forward(feature)
        last_inner = inner_topdown + inner_lateral
        results.insert(0, layer_block.forward(last_inner))
      }
    }
    results
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = null
    gradInput
  }
}

object FPN {
  def apply(in_channels_list: Array[Int], out_channels: Int): FPN
  = new FPN(in_channels_list, out_channels)
}
