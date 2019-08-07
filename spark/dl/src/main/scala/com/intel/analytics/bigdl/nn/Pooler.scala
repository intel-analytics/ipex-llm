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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class Pooler[T: ClassTag] (
  val resolution: Int,
  val scales: Array[T],
  val samplingRatio: Int
) (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T]{
  private val num_levels = scales.length
  private val poolers = new Array[RoiAlign[T]](num_levels)

  for (i <- 0 until num_levels) {
    poolers(i) = RoiAlign[T](scales(i), samplingRatio, resolution, resolution)
  }

  private val lvl_min = (-Math.log(scales(0).asInstanceOf[Double])/Math.log(2)).toInt
  private val lvl_max = (-Math.log(scales(num_levels - 1).asInstanceOf[Double])/Math.log(2)).toInt

  private def levelMapping(
    k_min: Int,
    k_max: Int,
    rois: Tensor[T],
    canonical_scale: Int = 224,
    canonical_level: Int = 4,
    eps: Float = 1e-6f
    ): Array[Int] = {
    val s0 = canonical_scale
    val lvl0 = canonical_level

    val target_lvls = new Array[Int](rois.size(1))
    for (i <- 1 to rois.size(1)) {
      val s = Math.sqrt(area(rois(i)).asInstanceOf[Double])
      var target_lvl = Math.floor(lvl0 + Math.log(s / s0 + eps) / Math.log(2))
      target_lvl = Math.min(Math.max(target_lvl, k_min), k_max)
      target_lvls(i) = (target_lvl - k_min).toInt
    }

    target_lvls
  }

  private def area(roi: Tensor[T]): T = {
    require(roi.size().length == 1 && roi.size(1) == 5,
      "ROI bounding box should be 1 dimensional and have 5 elements")
    val xlow = roi.valueAt(2)
    val ylow = roi.valueAt(3)
    val xhigh = roi.valueAt(4)
    val yhigh = roi.valueAt(5)

    val area = ev.times(ev.plus(ev.minus(xhigh, xlow), ev.fromType(1)),
      ev.plus(ev.minus(yhigh, ylow), ev.fromType(1)))
    area
  }

  override def updateOutput(input: Table): Tensor[T] = {
    val feature_maps = input[Table](1)
    val rois = input[Tensor[T]](2)

    val roi_levels = levelMapping(lvl_min, lvl_max, rois)
    val num_rois = rois.size(1)
    val num_channels = feature_maps.get[Tensor[T]]().get(1).size(2)

    output.resize(num_rois, num_channels, resolution, resolution)
      .fill(ev.fromType[Float](Float.MinValue))

    for (level <- 0 until num_levels) {
      val feature_per_level = feature_maps.get[Tensor[T]]().get(level + 1)
      val rois_ind_per_level = roi_levels.zipWithIndex.filter(_._1 == level).map(_._2)
      val num_rois_per_level = rois_ind_per_level.length

      val rois_per_level = Tensor[T](Array(num_rois_per_level, 5))
      for (i <- 0 until num_rois_per_level) {
        rois_per_level(i + 1) = rois(rois_ind_per_level(i))
      }

      val res = poolers(level).forward(T(feature_per_level, rois_per_level))
      for (i <- 0 until num_rois_per_level) {
        output(rois_ind_per_level(i) + 1) = res(i + 1)
      }
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    gradInput = null
    gradInput
  }
}

