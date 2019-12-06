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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect._

/**
 * Region of interest aligning (RoIAlign) for Mask-RCNN
 *
 * The RoIAlign uses average pooling on bilinear-interpolated sub-windows to convert
 * the features inside any valid region of interest into a small feature map with a
 * fixed spatial extent of pooledH * pooledW (e.g., 7 * 7).
 * An RoI is a rectangular window into a conv feature map.
 * Each RoI is defined by a four-tuple (x1, y1, x2, y2) that specifies its
 * top-left corner (x1, y1) and its bottom-right corner (x2, y2).
 * RoIAlign works by dividing the h * w RoI window into an pooledH * pooledW grid of
 * sub-windows of approximate size h/H * w/W. In each sub-window, compute exact values
 * of input features at four regularly sampled locations, and then do average pooling on
 * the values in each sub-window.
 * Pooling is applied independently to each feature map channel
 * @param spatialScale Spatial scale
 * @param samplingRatio Sampling ratio
 * @param pooledH spatial extent in height
 * @param pooledW spatial extent in width
 */
class RoiAlign[T: ClassTag] (
  val spatialScale: Float,
  val samplingRatio: Int,
  val pooledH: Int,
  val pooledW: Int,
  val mode: String = "avg",
  val aligned: Boolean = true
)(implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Tensor[T], T]{
  override def updateOutput(input: Activity): Tensor[T] = {
    if (classTag[T] == classTag[Float]) {
      val data = input.toTable[Tensor[Float]](1)
      val rois = input.toTable[Tensor[Float]](2)

      val num_rois = rois.size(1)
      val channels = data.size(2)
      val height = data.size(3)
      val width = data.size(4)

      output.resize(num_rois, channels, pooledH, pooledW)
        .fill(ev.fromType[Float](Float.MinValue))
      require(output.nElement() != 0, "Output contains no elements")

      val inputData = data.storage().array()
      val outputData = output.storage().array().asInstanceOf[Array[Float]]
      val roisFloat = rois.storage().array()

      poolOneRoiFloat(
        inputData,
        outputData,
        roisFloat,
        num_rois,
        channels,
        height,
        width,
        spatialScale)
    } else if (classTag[T] == classTag[Double]) {
      val data = input.toTable[Tensor[Double]](1)
      val rois = input.toTable[Tensor[Double]](2)

      val num_rois = rois.size(1)
      val channels = data.size(2)
      val height = data.size(3)
      val width = data.size(4)

      output.resize(num_rois, channels, pooledH, pooledW)
        .fill(ev.fromType[Double](Float.MinValue))
      require(output.nElement() != 0, "Output contains no elements")

      val inputData = data.storage().array()
      val outputData = output.storage().array().asInstanceOf[Array[Double]]
      val roisFloat = rois.storage().array()

      poolOneRoiDouble(
        inputData,
        outputData,
        roisFloat,
        num_rois,
        channels,
        height,
        width,
        spatialScale)
    } else {
      throw new IllegalArgumentException("currently only Double and Float types are supported")
    }

    output
  }


  private def bilinearInterpolateGradient(height: Int, width: Int, y: Float, x: Float)
    : (Float, Float, Float, Float, Int, Int, Int, Int) = {
    var w1: Float = 0.0f
    var w2: Float = 0.0f
    var w3: Float = 0.0f
    var w4: Float = 0.0f
    var x_low : Int = 0
    var x_high: Int = 0
    var y_low: Int = 0
    var y_high: Int = 0

    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
      // empty
      return (w1, w2, w3, w4, x_low, x_high, y_low, y_high)
    }

    var realY = if (y <= 0) 0 else y
    var realX = if (x <= 0) 0 else x
    y_low = realY.toInt
    x_low = realX.toInt

    if (y_low >= height - 1) {
      y_high = height - 1
      y_low = height - 1
      realY = y_low
    } else y_high = y_low + 1

    if (x_low >= width - 1) {
      x_high = width - 1
      x_low = width - 1
      realX = x_low
    } else x_high = x_low + 1

    val ly = realY - y_low
    val lx = realX - x_low
    val hy = 1.0 - ly
    val hx = 1.0 - lx

    w1 = (hy * hx).toFloat
    w2 = (hy * lx).toFloat
    w3 = (ly * hx).toFloat
    w4 = (ly * lx).toFloat

    return (w1, w2, w3, w4, x_low, x_high, y_low, y_high)
  }

  private def roiAlignBackward(
    nums: Int,
    gradOutputArr: Array[T],
    gradInputArr: Array[T],
    gradInputOffset: Int,
    rois: Array[T],
    channels: Int,
    height: Int,
    width: Int,
    pooled_height: Int,
    pooled_width: Int,
    sampling_ratio : Int,
    n_stride : Int,
    c_stride : Int,
    h_stride : Int,
    w_stride : Int,
    spatial_scale: Float) {
    val roi_cols = 4
    for (index <- 0 until nums) {
      val pw = index % pooled_width
      val ph = (index / pooled_width) % pooled_height
      val c = (index / pooled_width / pooled_height) % channels
      val n = index / pooled_width / pooled_height / channels
      val offset_rois = n * roi_cols

      val offset = if (aligned) 0.5f else 0.0f
      val roi_start_w = ev.toType[Float](rois(offset_rois)) * spatial_scale - offset
      val roi_start_h = ev.toType[Float](rois(offset_rois + 1)) * spatial_scale - offset
      val roi_end_w = ev.toType[Float](rois(offset_rois + 2)) * spatial_scale - offset
      val roi_end_h = ev.toType[Float](rois(offset_rois + 3)) * spatial_scale - offset

      var roi_width = roi_end_w - roi_start_w
      var roi_height = roi_end_h - roi_start_h

      if (aligned) {
        require(roi_width >= 0 && roi_height >= 0,
          s"ROIs in ROIAlign do not have non-negative size!" +
            s"But get ${roi_height} ${roi_width}")
      } else {
        roi_width = math.max(roi_width, 1.0f)
        roi_height = math.max(roi_height, 1.0f)
      }

      val bin_size_h = roi_height / pooled_height
      val bin_size_w = roi_width / pooled_width
      val output_offset = n * n_stride + c * c_stride
      val grad_output_value = gradOutputArr(output_offset + ph * h_stride + pw * w_stride)

      // We use roi_bin_grid to sample the grid and mimic integral
      val roi_bin_grid_h =
        if (sampling_ratio > 0) sampling_ratio else math.ceil(roi_height / pooled_height).toInt
      val roi_bin_grid_w =
        if (sampling_ratio > 0) sampling_ratio else math.ceil(roi_width / pooled_width).toInt

      // We do average (integral) pooling inside a bin
      val count = roi_bin_grid_h * roi_bin_grid_w

      for (iy <- 0 until roi_bin_grid_h) {
        val y = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
        for (ix <- 0 until roi_bin_grid_w) {
          val x = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w

          val (w1, w2, w3, w4, x_low, x_high, y_low, y_high) =
            bilinearInterpolateGradient(height, width, y.toFloat, x.toFloat)

          val g1 = ev.times(grad_output_value, ev.fromType(w1 / count))
          val g2 = ev.times(grad_output_value, ev.fromType(w2 / count))
          val g3 = ev.times(grad_output_value, ev.fromType(w3 / count))
          val g4 = ev.times(grad_output_value, ev.fromType(w4 / count))

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
            gradInputArr(gradInputOffset + y_low * width + x_low) =
              ev.plus(gradInputArr(gradInputOffset + y_low * width + x_low), g1)
            gradInputArr(gradInputOffset + y_low * width + x_high) =
              ev.plus(gradInputArr(gradInputOffset + y_low * width + x_high), g2)
            gradInputArr(gradInputOffset + y_high * width + x_low) =
              ev.plus(gradInputArr(gradInputOffset + y_high * width + x_low), g3)
            gradInputArr(gradInputOffset + y_high * width + x_high) =
              ev.plus(gradInputArr(gradInputOffset + y_high * width + x_high), g4)
          }
        }
      }
    }
  }

  override def updateGradInput(input: Activity, gradOutput: Tensor[T]): Activity = {
    require(mode == "avg", s"Only support backward for average mode, but get ${mode}")
    val data = input.toTable[Tensor[T]](1)
    val rois = input.toTable[Tensor[T]](2)
    val num_rois = rois.size(1)
    val channels = data.size(2)
    val height = data.size(3)
    val width = data.size(4)

    require(gradOutput.isContiguous(), "gradOutput should be contiguous")
    require(gradOutput.dim() == 4, s"gradOutput should be with 4 dims, but get ${gradOutput.dim()}")

    val n_stride = gradOutput.stride(1)
    val c_stride = gradOutput.stride(2)
    val h_stride = gradOutput.stride(3)
    val w_stride = gradOutput.stride(4)

    if (gradInput == null) gradInput = Tensor[T]()
    gradInput.toTensor[T].resize(channels, height, width)
    val gradInputArr = gradInput.toTensor[T].storage().array()
    val gradInputOffset = gradInput.toTensor[T].storageOffset() - 1

    roiAlignBackward(
      gradOutput.nElement(),
      gradOutputArr = gradOutput.asInstanceOf[Tensor[T]].storage().array(),
      gradInputArr = gradInputArr,
      gradInputOffset = 0,
      rois = rois.storage().array(),
      channels = channels,
      height = height,
      width = width,
      pooled_height = pooledH,
      pooled_width = pooledW,
      sampling_ratio = samplingRatio,
      n_stride = n_stride,
      c_stride = c_stride,
      h_stride = h_stride,
      w_stride = w_stride,
      spatial_scale = spatialScale)

    gradInput
  }

  private def poolOneRoiFloat(
    inputData: Array[Float],
    outputData: Array[Float],
    roisFloat: Array[Float],
    num_rois: Int,
    channels: Int,
    height: Int,
    width: Int,
    spatialScale: Float
  ): Unit = {
    val roi_cols = 4 // bbox has 4 elements

    for (n <- 0 until num_rois) {
      val index_n = n * channels * pooledW * pooledH
      val offset_rois = n * roi_cols
      val roi_batch_ind = 0 // bbox has 4 elements
      val alignedOffset = if (aligned) 0.5f else 0.0f
      val roi_start_w = roisFloat(offset_rois) * spatialScale  - alignedOffset
      val roi_start_h = roisFloat(offset_rois + 1) * spatialScale - alignedOffset
      val roi_end_w = roisFloat(offset_rois + 2) * spatialScale - alignedOffset
      val roi_end_h = roisFloat(offset_rois + 3) * spatialScale - alignedOffset
      var roi_width = roi_end_w - roi_start_w
      var roi_height = roi_end_h - roi_start_h

      if (aligned) {
        require(roi_width >= 0 && roi_height >= 0,
          "ROIs in ROIAlign cannot have non-negative size!")
      } else {
        roi_width = math.max(roi_width, 1.0f)
        roi_height = math.max(roi_height, 1.0f)
      }

      val bin_size_h = roi_height/ pooledH
      val bin_size_w = roi_width / pooledW

      val roi_bin_grid_h = if (samplingRatio > 0) {
        samplingRatio
      } else {
        Math.ceil(roi_height / pooledH).toInt
      }

      val roi_bin_grid_w = if (samplingRatio > 0) {
        samplingRatio
      } else {
        Math.ceil(roi_width / pooledW).toInt
      }

      val count: Float = math.max(roi_bin_grid_h * roi_bin_grid_w, 1.0f)

      val pre_cal = Tensor[Float](
        Array(pooledH * pooledW * roi_bin_grid_h * roi_bin_grid_w, 8))

      preCalcForBilinearInterpolateFloat(
        height,
        width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_cal
      )

      mode match {
        case "avg" =>
          for (c <- 0 until channels) {
            val index_n_c = index_n + c * pooledW * pooledH
            val offset_data = (roi_batch_ind * channels + c) * height * width
            var pre_calc_index: Int = 1

            for (ph <- 0 until pooledH) {
              for (pw <- 0 until pooledW) {
                val index = index_n_c + ph * pooledW + pw

                var output_val: Float = 0.0f
                for (iy <- 0 until roi_bin_grid_h) {
                  for (ix <- 0 until roi_bin_grid_w) {
                    val pc = pre_cal(pre_calc_index)
                    val pos1 = pc.valueAt(1).toInt
                    val pos2 = pc.valueAt(2).toInt
                    val pos3 = pc.valueAt(3).toInt
                    val pos4 = pc.valueAt(4).toInt
                    val w1 = pc.valueAt(5)
                    val w2 = pc.valueAt(6)
                    val w3 = pc.valueAt(7)
                    val w4 = pc.valueAt(8)

                    output_val = output_val + w1 * inputData(offset_data.toInt + pos1) +
                      w2 * inputData(offset_data.toInt + pos2) +
                      w3 * inputData(offset_data.toInt + pos3) +
                      w4 * inputData(offset_data.toInt + pos4)

                    pre_calc_index += 1
                  }
                }
                output_val /= count

                outputData(index) = output_val
              }
            }
          }
        case "max" =>
          for (c <- 0 until channels) {
            val index_n_c = index_n + c * pooledW * pooledH
            val offset_data = (roi_batch_ind * channels + c) * height * width
            var pre_calc_index: Int = 1

            for (ph <- 0 until pooledH) {
              for (pw <- 0 until pooledW) {
                val index = index_n_c + ph * pooledW + pw

                var output_val = Float.MinValue
                for (iy <- 0 until roi_bin_grid_h) {
                  for (ix <- 0 until roi_bin_grid_w) {
                    val pc = pre_cal(pre_calc_index)
                    val pos1 = pc.valueAt(1).toInt
                    val pos2 = pc.valueAt(2).toInt
                    val pos3 = pc.valueAt(3).toInt
                    val pos4 = pc.valueAt(4).toInt
                    val w1 = pc.valueAt(5)
                    val w2 = pc.valueAt(6)
                    val w3 = pc.valueAt(7)
                    val w4 = pc.valueAt(8)

                    val value = w1 * inputData(offset_data.toInt + pos1) +
                      w2 * inputData(offset_data.toInt + pos2) +
                      w3 * inputData(offset_data.toInt + pos3) +
                      w4 * inputData(offset_data.toInt + pos4)

                    if (value > output_val) {
                      output_val = value
                    }

                    pre_calc_index += 1
                  }
                }
                outputData(index) = output_val
              }
            }
          }
      }
    }
  }

  private def preCalcForBilinearInterpolateFloat(
    height: Int,
    width: Int,
    iy_upper: Int,
    ix_upper: Int,
    roi_start_h: Float,
    roi_start_w: Float,
    bin_size_h: Float,
    bin_size_w: Float,
    roi_bin_grid_h: Int,
    roi_bin_grid_w: Int,
    pre_cal: Tensor[Float]
  ) : Unit = {
    var pre_calc_index: Int = 1

    for (ph <- 0 until pooledH) {
      for (pw <- 0 until pooledW) {
        for (iy <- 0 until iy_upper) {
          val yy = roi_start_h + ph * bin_size_h + (iy + 0.5f) * bin_size_h / roi_bin_grid_h
          for (ix <- 0 until ix_upper) {
            val xx = roi_start_w + pw * bin_size_w + (ix + 0.5f) * bin_size_w / roi_bin_grid_w
            var x = xx
            var y = yy
            if (y < -1.0 || y > height || x < -1.0 || x > width) {
              pre_cal.setValue(pre_calc_index, 1, 0.0f) // pos1
              pre_cal.setValue(pre_calc_index, 2, 0.0f) // pos2
              pre_cal.setValue(pre_calc_index, 3, 0.0f) // pos3
              pre_cal.setValue(pre_calc_index, 4, 0.0f) // pos4
              pre_cal.setValue(pre_calc_index, 5, 0.0f) // w1
              pre_cal.setValue(pre_calc_index, 6, 0.0f) // w2
              pre_cal.setValue(pre_calc_index, 7, 0.0f) // w3
              pre_cal.setValue(pre_calc_index, 8, 0.0f) // w4
              pre_calc_index += 1
            } else {
              if (y <= 0) {
                y = 0
              }

              if (x <= 0) {
                x = 0
              }

              var y_low = y.toInt
              var x_low = x.toInt

              val y_high = if (y_low >= height - 1) {
                y_low = height -1
                y = y_low.toFloat
                y_low
              } else {
                y_low + 1
              }

              val x_high = if (x_low >= width - 1) {
                x_low = width -1
                x = x_low.toFloat
                x_low
              } else {
                x_low + 1
              }

              val ly = y - y_low
              val lx = x - x_low
              val hy = 1.0f - ly
              val hx = 1.0f - lx
              val w1 = hy * hx
              val w2 = hy * lx
              val w3 = ly * hx
              val w4 = ly * lx

              pre_cal.setValue(pre_calc_index, 1, y_low * width + x_low)
              pre_cal.setValue(pre_calc_index, 2, y_low * width + x_high)
              pre_cal.setValue(pre_calc_index, 3, y_high * width + x_low)
              pre_cal.setValue(pre_calc_index, 4, y_high * width + x_high)
              pre_cal.setValue(pre_calc_index, 5, w1)
              pre_cal.setValue(pre_calc_index, 6, w2)
              pre_cal.setValue(pre_calc_index, 7, w3)
              pre_cal.setValue(pre_calc_index, 8, w4)
              pre_calc_index += 1
            }
          }
        }
      }
    }
  }

  private def poolOneRoiDouble(
    inputData: Array[Double],
    outputData: Array[Double],
    roisDouble: Array[Double],
    num_rois: Int,
    channels: Int,
    height: Int,
    width: Int,
    spatialScale: Float
  ): Unit = {
    val roi_cols = 4 // bbox has 4 elements

    for (n <- 0 until num_rois) {
      val index_n = n * channels * pooledW * pooledH
      val offset_rois = n * roi_cols
      val roi_batch_ind = 0

      val alignedOffset = if (aligned) 0.5f else 0.0f
      val roi_start_w = roisDouble(offset_rois) * spatialScale  - alignedOffset
      val roi_start_h = roisDouble(offset_rois + 1) * spatialScale - alignedOffset
      val roi_end_w = roisDouble(offset_rois + 2) * spatialScale - alignedOffset
      val roi_end_h = roisDouble(offset_rois + 3) * spatialScale - alignedOffset

      var roi_width = roi_end_w - roi_start_w
      var roi_height = roi_end_h - roi_start_h
      if (aligned) {
        require(roi_width >= 0 && roi_height >= 0,
          "ROIs in ROIAlign cannot have non-negative size!")
      } else {
        roi_width = math.max(roi_width, 1.0f)
        roi_height = math.max(roi_height, 1.0f)
      }
      val bin_size_h = roi_height/ pooledH
      val bin_size_w = roi_width / pooledW

      val roi_bin_grid_h = if (samplingRatio > 0) {
        samplingRatio
      } else {
        Math.ceil(roi_height / pooledH).toInt
      }

      val roi_bin_grid_w = if (samplingRatio > 0) {
        samplingRatio
      } else {
        Math.ceil(roi_width / pooledW).toInt
      }

      val count: Double = math.max(roi_bin_grid_h * roi_bin_grid_w, 1.0f)
      val pre_cal = Tensor[Double](
        Array(pooledH * pooledW * roi_bin_grid_h * roi_bin_grid_w, 8))

      preCalcForBilinearInterpolateDouble(
        height,
        width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_cal
      )
      mode match {
        case "avg" =>
          for (c <- 0 until channels) {
            val index_n_c = index_n + c * pooledW * pooledH
            val offset_data = (roi_batch_ind * channels + c) * height * width
            var pre_calc_index: Int = 1

            for (ph <- 0 until pooledH) {
              for (pw <- 0 until pooledW) {
                val index = index_n_c + ph * pooledW + pw

                var output_val: Double = 0.0
                for (iy <- 0 until roi_bin_grid_h) {
                  for (ix <- 0 until roi_bin_grid_w) {
                    val pc = pre_cal(pre_calc_index)
                    val pos1 = pc.valueAt(1).toInt
                    val pos2 = pc.valueAt(2).toInt
                    val pos3 = pc.valueAt(3).toInt
                    val pos4 = pc.valueAt(4).toInt
                    val w1 = pc.valueAt(5)
                    val w2 = pc.valueAt(6)
                    val w3 = pc.valueAt(7)
                    val w4 = pc.valueAt(8)

                    output_val = output_val +  w1 * inputData(offset_data.toInt + pos1) +
                      w2 * inputData(offset_data.toInt + pos2) +
                      w3 * inputData(offset_data.toInt + pos3) +
                      w4 * inputData(offset_data.toInt + pos4)

                    pre_calc_index += 1
                  }
                }
                output_val /= count

                outputData(index) = output_val
              }
            }
          }
        case "max" =>
          for (c <- 0 until channels) {
            val index_n_c = index_n + c * pooledW * pooledH
            val offset_data = (roi_batch_ind * channels + c) * height * width
            var pre_calc_index: Int = 1

            for (ph <- 0 until pooledH) {
              for (pw <- 0 until pooledW) {
                val index = index_n_c + ph * pooledW + pw

                var output_val = Double.MinValue
                for (iy <- 0 until roi_bin_grid_h) {
                  for (ix <- 0 until roi_bin_grid_w) {
                    val pc = pre_cal(pre_calc_index)
                    val pos1 = pc.valueAt(1).toInt
                    val pos2 = pc.valueAt(2).toInt
                    val pos3 = pc.valueAt(3).toInt
                    val pos4 = pc.valueAt(4).toInt
                    val w1 = pc.valueAt(5)
                    val w2 = pc.valueAt(6)
                    val w3 = pc.valueAt(7)
                    val w4 = pc.valueAt(8)

                    val value = w1 * inputData(offset_data.toInt + pos1) +
                      w2 * inputData(offset_data.toInt + pos2) +
                      w3 * inputData(offset_data.toInt + pos3) +
                      w4 * inputData(offset_data.toInt + pos4)

                    if (value > output_val) {
                      output_val = value
                    }

                    pre_calc_index += 1
                  }
                }
                outputData(index) = output_val
              }
            }
          }
      }
    }
  }

  private def preCalcForBilinearInterpolateDouble(
    height: Int,
    width: Int,
    iy_upper: Int,
    ix_upper: Int,
    roi_start_h: Double,
    roi_start_w: Double,
    bin_size_h: Double,
    bin_size_w: Double,
    roi_bin_grid_h: Int,
    roi_bin_grid_w: Int,
    pre_cal: Tensor[Double]
  ) : Unit = {
    var pre_calc_index: Int = 1

    for (ph <- 0 until pooledH) {
      for (pw <- 0 until pooledW) {
        for (iy <- 0 until iy_upper) {
          val yy = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
          for (ix <- 0 until ix_upper) {
            val xx = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w
            var x = xx
            var y = yy
            if (y < -1.0 || y > height || x < -1.0 || x > width) {
              pre_cal.setValue(pre_calc_index, 1, 0.0) // pos1
              pre_cal.setValue(pre_calc_index, 2, 0.0) // pos2
              pre_cal.setValue(pre_calc_index, 3, 0.0) // pos3
              pre_cal.setValue(pre_calc_index, 4, 0.0) // pos4
              pre_cal.setValue(pre_calc_index, 5, 0.0) // w1
              pre_cal.setValue(pre_calc_index, 6, 0.0) // w2
              pre_cal.setValue(pre_calc_index, 7, 0.0) // w3
              pre_cal.setValue(pre_calc_index, 8, 0.0) // w4
              pre_calc_index += 1
            }

            else {
              if (y <= 0) {
                y = 0
              }

              if (x <= 0) {
                x = 0
              }

              var y_low = y.toInt
              var x_low = x.toInt

              val y_high = if (y_low >= height - 1) {
                y_low = height -1
                y = y_low.toDouble
                y_low
              } else {
                y_low + 1
              }

              val x_high = if (x_low >= width - 1) {
                x_low = width -1
                x = x_low.toDouble
                x_low
              } else {
                x_low + 1
              }

              val ly = y - y_low
              val lx = x - x_low
              val hy = 1.0f - ly
              val hx = 1.0f - lx
              val w1 = hy * hx
              val w2 = hy * lx
              val w3 = ly * hx
              val w4 = ly * lx

              pre_cal.setValue(pre_calc_index, 1, y_low * width + x_low)
              pre_cal.setValue(pre_calc_index, 2, y_low * width + x_high)
              pre_cal.setValue(pre_calc_index, 3, y_high * width + x_low)
              pre_cal.setValue(pre_calc_index, 4, y_high * width + x_high)
              pre_cal.setValue(pre_calc_index, 5, w1)
              pre_cal.setValue(pre_calc_index, 6, w2)
              pre_cal.setValue(pre_calc_index, 7, w3)
              pre_cal.setValue(pre_calc_index, 8, w4)
              pre_calc_index += 1
            }
          }
        }
      }
    }
  }

  override def toString: String = "nn.RoiAlign"

  override def clearState(): this.type = {
    super.clearState()
    this
  }
}

object RoiAlign {
  def apply[@specialized(Float, Double) T: ClassTag](
    spatialScale: Float,
    samplingRatio: Int,
    pooledH: Int,
    pooledW: Int,
    mode: String = "avg",
    aligned: Boolean = true
  ) (implicit ev: TensorNumeric[T]): RoiAlign[T] =
    new RoiAlign[T](spatialScale, samplingRatio, pooledH, pooledW, mode, aligned)
}
