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
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Feature Pyramid Network.
 * @param inChannels        number of channels of feature maps
 * @param outChannels       number of channels of FPN output
 * @param topBlocks         Top Blocks option
 *                          Extra operation to be performed on the smallest
 *                          resolution FPN output, whose result is appended
 *                          to the result list
 *                          0 for null,
 *                          1 for using max pooling on the last level
 *                          2 for extra layers P6 and P7 in RetinaNet
 * @param inChannelsOfP6P7    number of input channels of P6 P7 in RetinaNet
 * @param outChannelsOfP6P7   number of output channels of P6 P7 in RetinaNet
 */

class FPN[T : ClassTag](
  val inChannels: Array[Int],
  val outChannels: Int,
  val topBlocks: Int = 0,
  val inChannelsOfP6P7: Int = 0,
  val outChannelsOfP6P7: Int = 0
)
  (implicit ev: TensorNumeric[T])
  extends BaseModule[T]{
  override def buildModel(): Module[T] = {
    val featureMapsNum = inChannels.length
    val innerBlockModules = new Array[SpatialConvolution[T]](featureMapsNum)
    val layerBlockModules = new Array[SpatialConvolution[T]](featureMapsNum)

    for (i <- 0 to featureMapsNum - 1) {
      if (inChannels(i) != 0) {
        val innerBlockModule =
          SpatialConvolution[T](inChannels(i), outChannels, 1, 1, 1, 1)
            .setName(s"fpn_inner${i + 1}")
        val layerBlockModule =
          SpatialConvolution[T](outChannels, outChannels, 3, 3, 1, 1, 1, 1)
            .setName(s"fpn_layer${i + 1}")
        innerBlockModules(i) = innerBlockModule
        layerBlockModules(i) = layerBlockModule
      }
    }

    val inputs = new Array[ModuleNode[T]](featureMapsNum)
    for (i <- 0 to featureMapsNum - 1) {
      inputs(i) = Input[T]()
    }

    val innerBlocks = new Array[ModuleNode[T]](featureMapsNum)
    for (i <- 0 to featureMapsNum - 1) {
      innerBlocks(i) = innerBlockModules(i).inputs(inputs(i))
    }

    val results = new Array[ModuleNode[T]](featureMapsNum + topBlocks)
    var count = results.length - 1 - topBlocks

    var lastInner = innerBlocks(featureMapsNum - 1)
    results(count) = layerBlockModules(featureMapsNum - 1).inputs(lastInner)

    for(i <- featureMapsNum - 2 to 0 by -1) {
      val layerBlock = layerBlockModules(i)
      if (layerBlock != null) {
        val innerTopDown = UpSampling2D[T](Array(2, 2)).inputs(lastInner)
        val innerLateral = innerBlocks(i)
        lastInner = CAddTable[T]().setName(s"number_${i}_${featureMapsNum}")
          .inputs(innerLateral, innerTopDown)
        count -= 1
        results(count) = layerBlock.inputs(lastInner)
      }
    }

    if (topBlocks == 1) {
      results(results.length - 1) = SpatialMaxPooling(1, 1, 2, 2)
        .inputs(results(featureMapsNum - 1))
    }

    if (topBlocks == 2) {
      val p6_module = SpatialConvolution[T](inChannelsOfP6P7, outChannelsOfP6P7, 3, 3, 2, 2, 1, 1)
      val p7_module = SpatialConvolution[T](outChannelsOfP6P7, outChannelsOfP6P7, 3, 3, 2, 2, 1, 1)
      results(results.length - 2) = if (inChannelsOfP6P7 == outChannelsOfP6P7) {
        p6_module.inputs(results(featureMapsNum - 1))
      } else {
        p6_module.inputs(inputs(featureMapsNum - 1))
      }
      results(results.length - 1) = p7_module.inputs(ReLU[T]().inputs(results(results.length - 2)))
    }

    Graph(inputs, results)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    throw new UnsupportedOperationException("Not support backward propagation")
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[FPN[T]]

  override def equals(other: Any): Boolean = other match {
    case that: FPN[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        inChannels.deep == that.inChannels.deep &&
        outChannels == that.outChannels
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), inChannels, outChannels)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def reset(): Unit = {
    super.reset()
    model.reset()
  }

  override def toString: String = s"FPN($outChannels)"
}

object FPN {
  def apply[@specialized(Float, Double) T: ClassTag](
    inChannels: Array[Int],
    outChannels: Int,
    topBlocks: Int = 0,
    inChannelsOfP6P7: Int = 0,
    outChannelsOfP6P7: Int = 0
  )(implicit ev: TensorNumeric[T]): FPN[T] = {
    new FPN[T](inChannels, outChannels, topBlocks, inChannelsOfP6P7, outChannelsOfP6P7)
  }
}
