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
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect._

class RoiAlign[T: ClassTag] (
  val output_size: Array[Int],
  val spatio_scale: T,
  val sampling_ratio: T,
  val pooled_height: Int,
  val pooled_width: Int
) (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T]{
  private var pre_cal: Tensor[Tensor[T]] = _

  override def updateOutput(input: Table): Tensor[T] = {
    val data = input[Tensor[T]](1)
    val rois = input[Tensor[T]](2)

    val num_rois = rois.size(0)
    val channels = data.size(1)
    val height = data.size(2)
    val width = data.size(3)

    output.resize(num_rois, channels, pooled_height, pooled_width)
      .fill(ev.fromType[Double](Double.MinValue))

    val output_size = num_rois * channels * pooled_height * pooled_width

    require(output.nElement() != 0, "Output contains no elements")

    val roi_cols = 5

    val tensor = Tensor[T](Array(2, 2))

    tensor
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    gradInput = null
    gradInput
  }

  private def preCalcForBilinearInterpolateFloat(
    height: Int,
    width: Int,
    pooled_height: Int,
    pooled_width: Int,
    iy_upper: Int,
    ix_upper: Int,
    roi_start_h: Float,
    roi_start_w: Float,
    bin_size_h: Float,
    bin_size_w: Float,
    roi_bin_grid_h: Int,
    roi_bin_grid_w: Int
  ) : Unit = {
    var pre_calc_index = 0
    pre_cal.resize(pooled_height * pooled_width * iy_upper * ix_upper)

    for (ph <- 0 until pooled_height) {
      for (pw <- 0 until pooled_width) {
        for (iy <- 0 until iy_upper) {
          val yy = roi_start_h + ph * bin_size_h + (iy + 0.5F) * bin_size_h / roi_bin_grid_h
          for (ix <- 0 until ix_upper) {
            val xx = roi_start_w + pw * bin_size_w + (ix + 0.5F) * bin_size_w / roi_bin_grid_w
            var x = xx
            var y = yy
            if (y < -1.0 || y > height || x < -1.0 || x > width) {
              val pc = Tensor[Float](8)
              pc.setValue(1, 0) // pos1
              pc.setValue(2, 0) // pos2
              pc.setValue(3, 0) // pos3
              pc.setValue(4, 0) // pos4
              pc.setValue(5, 0) // w1
              pc.setValue(6, 0) // w2
              pc.setValue(7, 0) // w3
              pc.setValue(8, 0) // w4
              pre_cal(pre_calc_index) = pc
              pre_calc_index += 1
            }

            else {
              if (y <= 0) {
                y = 0
              }

              if (x <= 0) {
                x = 0
              }

              var y_low = y.asInstanceOf[Int]
              var x_low = x.asInstanceOf[Int]

              val y_high = if (y_low >= height - 1) {
                y_low = height -1
                y = y_low.asInstanceOf[Float]
                y_low
              } else {
                y_low + 1
              }

              val x_high = if (x_low >= width - 1) {
                x_low = width -1
                x = x_low.asInstanceOf[Float]
                x_low
              } else {
                x_low + 1
              }

              val ly = y - y_low
              val lx = x - x_low
              val hy = 1.0F - ly
              val hx = 1.0F - lx
              val w1 = hy * hx
              val w2 = hy * lx
              val w3 = ly * hx
              val w4 = ly * lx

              val pc = Tensor[Float](8)
              pc.setValue(1, y_low * width + x_low)
              pc.setValue(2, y_low * width + x_high)
              pc.setValue(3, y_high * width + x_low)
              pc.setValue(4, y_high * width + x_high)
              pc.setValue(5, w1)
              pc.setValue(6, w2)
              pc.setValue(7, w3)
              pc.setValue(8, w4)
              pre_cal(pre_calc_index) = pc
              pre_calc_index += 1
            }
          }
        }
      }
    }
  }

  private def preCalcForBilinearInterpolateDouble(
    height: Int,
    width: Int,
    pooled_height: Int,
    pooled_width: Int,
    iy_upper: Int,
    ix_upper: Int,
    roi_start_h: Double,
    roi_start_w: Double,
    bin_size_h: Double,
    bin_size_w: Double,
    roi_bin_grid_h: Int,
    roi_bin_grid_w: Int
  ) : Unit = {
    var pre_calc_index = 0
    pre_cal.resize(pooled_height * pooled_width * iy_upper * ix_upper)

    for (ph <- 0 until pooled_height) {
      for (pw <- 0 until pooled_width) {
        for (iy <- 0 until iy_upper) {
          val yy = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
          for (ix <- 0 until ix_upper) {
            val xx = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w
            var x = xx
            var y = yy
            if (y < -1.0 || y > height || x < -1.0 || x > width) {
              val pc = Tensor[Double](8)
              pc.setValue(1, 0) // pos1
              pc.setValue(2, 0) // pos2
              pc.setValue(3, 0) // pos3
              pc.setValue(4, 0) // pos4
              pc.setValue(5, 0) // w1
              pc.setValue(6, 0) // w2
              pc.setValue(7, 0) // w3
              pc.setValue(8, 0) // w4
              pre_cal(pre_calc_index) = pc
              pre_calc_index += 1
            }

            else {
              if (y <= 0) {
                y = 0
              }

              if (x <= 0) {
                x = 0
              }

              var y_low = y.asInstanceOf[Int]
              var x_low = x.asInstanceOf[Int]

              val y_high = if (y_low >= height - 1) {
                y_low = height -1
                y = y_low.asInstanceOf[Double]
                y_low
              } else {
                y_low + 1
              }

              val x_high = if (x_low >= width - 1) {
                x_low = width -1
                x = x_low.asInstanceOf[Double]
                x_low
              } else {
                x_low + 1
              }

              val ly = y - y_low
              val lx = x - x_low
              val hy = 1.0F - ly
              val hx = 1.0F - lx
              val w1 = hy * hx
              val w2 = hy * lx
              val w3 = ly * hx
              val w4 = ly * lx

              val pc = Tensor[Double](8)
              pc.setValue(1, y_low * width + x_low)
              pc.setValue(2, y_low * width + x_high)
              pc.setValue(3, y_high * width + x_low)
              pc.setValue(4, y_high * width + x_high)
              pc.setValue(5, w1)
              pc.setValue(6, w2)
              pc.setValue(7, w3)
              pc.setValue(8, w4)
              pre_cal(pre_calc_index) = pc
              pre_calc_index += 1
            }
          }
        }
      }
    }
  }
}

object RoiAlign {
  def apply[@specialized(Float, Double) T: ClassTag](
    output_size: Array[Int],
    spatio_scale: T,
    sampling_ratio: T,
    pooled_height: Int,
    pooled_width: Int) (implicit ev: TensorNumeric[T]): RoiAlign[T] =
    new RoiAlign[T](output_size, spatio_scale, sampling_ratio, pooled_height, pooled_width)
}
