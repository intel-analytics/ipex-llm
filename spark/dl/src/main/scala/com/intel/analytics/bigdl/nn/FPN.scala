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
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Feature Pyramid Network.
 * @param in_channels_list number of channels of feature maps
 * @param out_channels number of channels of FPN output
 */

class FPN[T : ClassTag](
  val in_channels_list: Array[Int],
  val out_channels: Int
)
  (implicit ev: TensorNumeric[T])
  extends BaseModule[T]{
  override def buildModel(): Module[T] = {
    val num_feature_maps = in_channels_list.length
    val inner_blocks_modules = new Array[SpatialConvolution[T]](num_feature_maps)
    val layer_blocks_modules = new Array[SpatialConvolution[T]](num_feature_maps)

    for (i <- 0 to num_feature_maps - 1) {
      if (in_channels_list(i) != 0) {
        val inner_block_module =
          SpatialConvolution[T](in_channels_list(i), out_channels, 1, 1, 1, 1)
        val layer_block_module =
          SpatialConvolution[T](out_channels, out_channels, 3, 3, 1, 1, 1, 1)
        inner_blocks_modules(i) = inner_block_module
        layer_blocks_modules(i) = layer_block_module
      }
    }

    val input = Input[T]()
    val inner_conv = ParallelTable[T]()
    for (i <- 0 to num_feature_maps - 1) {
      inner_conv.add(inner_blocks_modules(i))
    }
    val inner_block = inner_conv.inputs(input)

    var count = 0
    var results = new Array[ModuleNode[T]](num_feature_maps)
    var last_inner = SelectTable[T](num_feature_maps).inputs(inner_block)
    results(count) = layer_blocks_modules(num_feature_maps - 1).inputs(last_inner)

    for(i <- num_feature_maps - 1 to 1 by -1) {
      val layer_block = layer_blocks_modules(i - 1)
      if (layer_block != null) {
        val inner_topdown = UpSampling2D[T](Array(2, 2)).inputs(last_inner)
        val inner_lateral = SelectTable[T](i).inputs(inner_block)
        last_inner = CAddTable[T]().inputs(inner_lateral, inner_topdown)
        count += 1
        results(count) = layer_block.inputs(last_inner)
      }
    }

    Graph(Array(input), results)
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[FPN[T]]

  override def equals(other: Any): Boolean = other match {
    case that: FPN[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        in_channels_list.deep == that.in_channels_list.deep &&
        out_channels == that.out_channels
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), in_channels_list, out_channels)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def reset(): Unit = {
    super.reset()
    model.reset()
  }

  override def toString: String = s"FPN($out_channels)"
}

object FPN {
  def apply[@specialized(Float, Double) T: ClassTag](
    in_channels_list: Array[Int],
    out_channels: Int
  )(implicit ev: TensorNumeric[T]): FPN[T] = {
    new FPN[T](in_channels_list, out_channels)
  }
}
