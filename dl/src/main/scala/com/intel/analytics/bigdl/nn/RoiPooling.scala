/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect._

/**
 * Region of interest pooling
 * @param pooledW
 * @param pooledH
 * @param spatialScale
 * @tparam T Numeric type. Only support float/double now
 */
class RoiPooling[T: ClassTag](val pooledW: Int, val pooledH: Int, val spatialScale: T)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  @transient private var argmax: Tensor[T] = _
  @transient private var argmaxData: Array[T] = _
  @transient private var inputData: Array[T] = _
  @transient private var outputData: Array[T] = _
  @transient private var gradInputTensor: Tensor[T] = _

  override def updateOutput(input: Table): Tensor[T] = {
    assert(input.length() == 2, "there must have two tensors in the table")

    val data = input[Tensor[T]](1) // Input data to ROIPooling
    val rois = input[Tensor[T]](2) // Input label to ROIPooling

    assert(rois.size().length > 1 && rois.size(2) == 5, "roi input shape should be (R, 5)")

    output.resize(rois.size(1), data.size(2), pooledH, pooledW)
      .fill(ev.fromType[Double](-Double.MaxValue))
    outputData = output.storage().array()
    if (argmax == null) {
      argmax = Tensor[T]()
    }
    argmax.resizeAs(output).fill(ev.fromType(-1))
    argmaxData = argmax.storage().array()
    inputData = data.storage().array()
    val dataOffset = offset(0, 1, sizes = data.size())
    val argmaxOffset = offset(0, 1, sizes = argmax.size())
    val outputOffset = offset(0, 1, sizes = output.size())
    val roisOffset = offset(1, sizes = rois.size())
    var n = 0
    while (n < rois.size(1)) {
      poolOneRoi(n, rois(n + 1), data, dataOffset, argmaxOffset, outputOffset, roisOffset)
      n += 1
    }
    output
  }

  def poolOneRoi(n: Int, roi: Tensor[T], data: Tensor[T],
    dataOffset: Int, argmaxOffset: Int, outputOffset: Int, roisOffset: Int): Unit = {
    val roiBatchInd = ev.toType[Int](roi.valueAt(1))
    def scaleRoi(ind: Int) = Math.round(ev.toType[Double](roi.valueAt(ind))
      * ev.toType[Double](spatialScale)).toInt
    val roiStartW = scaleRoi(2)
    val roiStartH = scaleRoi(3)

    require(roiBatchInd >= 0 && data.size(1) > roiBatchInd)

    val binSizeH = Math.max(scaleRoi(5) - roiStartH + 1, 1f) / pooledH
    val binSizeW = Math.max(scaleRoi(4) - roiStartW + 1, 1f) / pooledW
    var batchDataIndex = offset(roiBatchInd, sizes = data.size())

    var c = 0
    while (c < data.size(2)) {
      var ph = 0
      val outputDataIndex = outputOffset * (n * data.size(2) + c)
      val argmaxIndex = argmaxOffset * (n * data.size(2) + c)
      while (ph < pooledH) {
        var pw = 0
        while (pw < pooledW) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          val hstart = Math.min(Math.max(Math.floor(ph * binSizeH).toInt + roiStartH, 0),
            data.size(3))
          val hend = Math.min(Math.max(Math.ceil((ph + 1) * binSizeH).toInt + roiStartH, 0),
            data.size(3))
          val wstart = Math.min(Math.max(Math.floor(pw * binSizeW).toInt + roiStartW, 0),
            data.size(4))
          val wend = Math.min(Math.max(Math.ceil((pw + 1) * binSizeW).toInt + roiStartW, 0),
            data.size(4))

          val poolIndex = ph * pooledW + pw
          if ((hend <= hstart) || (wend <= wstart)) {
            outputData(outputDataIndex + poolIndex) = ev.fromType(0)
            argmaxData(argmaxIndex + poolIndex) = ev.fromType(-1)
          } else {
            var h = hstart
            while (h < hend) {
              var w = wstart
              val hi = h * data.size(4)
              while (w < wend) {
                val index = hi + w
                if (ev.isGreater(inputData(batchDataIndex + index),
                  outputData(outputDataIndex + poolIndex))) {
                  outputData(outputDataIndex + poolIndex) = inputData(batchDataIndex + index)
                  argmaxData(argmaxIndex + poolIndex) = ev.fromType(index)
                }
                w += 1
              }
              h += 1
            }
          }
          pw += 1
        }
        ph += 1
      }
      // Increment all data pointers by one channel
      c += 1
      batchDataIndex += dataOffset
    }
  }


  def offset(n: Int, c: Int = 0, h: Int = 0, w: Int = 0, sizes: Array[Int]): Int = {
    assert(sizes.length == 2 || sizes.length >= 4)
    if (sizes.length == 2) ((n * sizes(1) + c) + h) + w
    else ((n * sizes(1) + c) * sizes(2) + h) * sizes(3) + w
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val data = input[Tensor[T]](1)
    val roisData = input[Tensor[T]](2).storage().array()
    val argmaxData = argmax.storage().array()
    val numRois = output.size(1)
    if (gradInputTensor == null) {
      gradInputTensor = Tensor[T]()
      gradInput.insert(gradInputTensor)
    }
    val gradInputData = gradInputTensor.resizeAs(data).contiguous().zero().storage().array()
    val gradOutputData = gradOutput.storage().array()
    var roiN = 0
    var c = 0
    var ph = 0
    var pw = 0
    // Accumulate gradient over all ROIs
    while (roiN < numRois) {
      val roiBatchInd = roisData(roiN * 5)
      // Accumulate gradients over each bin in this ROI
      c = 0
      while (c < data.size(2)) {
        ph = 0
        while (ph < pooledH) {
          pw = 0
          while (pw < pooledW) {
            val outputOffset = ((roiN * data.size(2) + c) * pooledH + ph) * pooledW + pw
            val argmaxIndex = argmaxData(outputOffset)
            if (ev.toType[Double](argmaxIndex) >= 0) {
              val inputOffset = (ev.toType[Int](roiBatchInd) * data.size(2)
                + c) * data.size(3) * data.size(4) + ev.toType[Int](argmaxIndex)
              gradInputData(inputOffset) =
                ev.plus(gradInputData(inputOffset), gradOutputData(outputOffset))
            }
            pw += 1
          }
          ph += 1
        }
        c += 1
      }
      roiN += 1
    }
    gradInput
  }

  override def toString: String = "nn.RoiPooling"
}

object RoiPooling {
  def apply[@specialized(Float, Double) T: ClassTag](
    pooled_w: Int, pooled_h: Int, spatial_scale: T)(implicit ev: TensorNumeric[T]): RoiPooling[T] =
    new RoiPooling[T](pooled_w, pooled_h, spatial_scale)
}