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

import scala.reflect._

/**
 * Pooler selects the feature map which matches the size of RoI for RoIAlign
 * @param resolution The resolution of pooled feature maps. Height equals width.
 * @param scales Spatial scales of each feature map
 * @param samplingRatio Sampling ratio
 */

class Pooler[T: ClassTag] (
  val resolution: Int,
  val scales: Array[Float],
  val samplingRatio: Int
) (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T]{
  private val num_levels = scales.length
  private val poolers = new Array[RoiAlign[T]](num_levels)

  for (i <- 0 until num_levels) {
    poolers(i) = RoiAlign[T](scales(i), samplingRatio, resolution, resolution)
  }

  private val lvl_min = if (classTag[T] == classTag[Float]) {
    (-Math.log(scales(0))/Math.log(2.0)).toInt
  } else if (classTag[T] == classTag[Double]) {
    (-Math.log(scales(0))/Math.log(2.0)).toInt
  } else {
    throw new IllegalArgumentException("currently only Double and Float types are supported")
  }

  private val lvl_max = if (classTag[T] == classTag[Float]) {
    (-Math.log(scales(num_levels - 1))/Math.log(2.0)).toInt
  } else if (classTag[T] == classTag[Double]) {
    (-Math.log(scales(num_levels - 1))/Math.log(2.0)).toInt
  } else {
    throw new IllegalArgumentException("currently only Double and Float types are supported")
  }

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
      val a = if (classTag[T] == classTag[Float]) {
        area(rois(i)).asInstanceOf[Float].toDouble
      } else if (classTag[T] == classTag[Double]) {
        area(rois(i)).asInstanceOf[Double]
      } else {
        throw new IllegalArgumentException("currently only Double and Float types are supported")
      }

      val s = Math.sqrt(a)
      var target_lvl = Math.floor(lvl0 + Math.log(s / s0 + eps) / Math.log(2))
      target_lvl = Math.min(Math.max(target_lvl, k_min), k_max)
      target_lvls(i - 1) = (target_lvl - k_min).toInt
    }

    target_lvls
  }

  private def area(roi: Tensor[T]): T = {
    require(roi.size().length == 1 && roi.size(1) == 4,
      s"ROI bounding box should be 1 dimensional and have 4 elements " +
        s"(xlow, ylow, xhigh, yhigh)")
    val xlow = roi.valueAt(1)
    val ylow = roi.valueAt(2)
    val xhigh = roi.valueAt(3)
    val yhigh = roi.valueAt(4)

    val area = ev.times(ev.plus(ev.minus(xhigh, xlow), ev.fromType(1)),
      ev.plus(ev.minus(yhigh, ylow), ev.fromType(1)))
    area
  }

  override def updateOutput(input: Table): Tensor[T] = {
    val featureMaps = input[Table](1)
    val roiBatch = if (input(2).isInstanceOf[Tensor[T]]) {
      T(input[Tensor[T]](2))
    } else { // for batch support
      input[Table](2)
    }

    val batchSize = featureMaps.get[Tensor[Float]](1).get.size(1)
    var totalNum = 0
    val num_channels = featureMaps.get[Tensor[T]](1).get.size(2)
    val out = T()
    for (i <- 0 to batchSize - 1) {
      val rois = roiBatch[Tensor[T]](i + 1)

      val roi_levels = levelMapping(lvl_min, lvl_max, rois)
      val num_rois = rois.size(1)
      totalNum += num_rois

      if (!out.contains(i + 1)) out(i + 1) = Tensor[T]()
      val outROI = out[Tensor[T]](i + 1)
      outROI.resize(num_rois, num_channels, resolution, resolution)
        .fill(ev.fromType[Float](Float.MinValue))

      for (level <- 0 until num_levels) {
        val tmp = featureMaps.get[Tensor[T]](level + 1).get.narrow(1, i + 1, 1)
        val feature_per_level = Tensor[T]().resizeAs(tmp).copy(tmp)
        val rois_ind_per_level = roi_levels.zipWithIndex.filter(_._1 == level).map(_._2)
        val num_rois_per_level = rois_ind_per_level.length

        if (num_rois_per_level > 0) {
          val rois_per_level = Tensor[T](Array(num_rois_per_level, 4)) // bbox has 4 elements
          for (i <- 0 until num_rois_per_level) {
            rois_per_level(i + 1) = rois(rois_ind_per_level(i) + 1)
          }

          val res = poolers(level).forward(T(feature_per_level, rois_per_level))
          for (i <- 0 until num_rois_per_level) {
            outROI(rois_ind_per_level(i) + 1) = res(i + 1)
          }
        }
      }
    }

    // merge to one tensor
    output.resize(totalNum, num_channels, resolution, resolution)
    var start = 1
    for (i <- 0 to batchSize - 1) {
      val tmp = out[Tensor[T]](i + 1)
      val length = tmp.size(1)
      if (length > 0) output.narrow(1, start, length).copy(tmp)
      start += length
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    throw new UnsupportedOperationException("Not support backward propagation")
  }

  override def toString: String = "nn.Pooler"

  override def clearState(): this.type = {
    super.clearState()
    for (i <- 0 until num_levels) {
      poolers(i).clearState()
    }
    this
  }
}

object Pooler {
  def apply[@specialized(Float, Double) T: ClassTag](
    resolution: Int,
    scales: Array[Float],
    samplingRatio: Int) (implicit ev: TensorNumeric[T]): Pooler[T] =
    new Pooler[T](resolution, scales, samplingRatio)
}

