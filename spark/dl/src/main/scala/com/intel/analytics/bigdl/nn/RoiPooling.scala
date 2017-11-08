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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect._

/**
 * Region of interest pooling
 * The RoIPooling uses max pooling to convert the features inside any valid region of interest
 * into a small feature map with a fixed spatial extent of pooledH × pooledW (e.g., 7 × 7)
 * an RoI is a rectangular window into a conv feature map.
 * Each RoI is defined by a four-tuple (x1, y1, x2, y2) that specifies its
 * top-left corner (x1, y1) and its bottom-right corner (x2, y2).
 * RoI max pooling works by dividing the h × w RoI window into an pooledH × pooledW grid of
 * sub-windows of approximate size h/H × w/W and then max-pooling the values in each sub-window
 * into the corresponding output grid cell.
 * Pooling is applied independently to each feature map channel
 * @param pooledW      spatial extent in width
 * @param pooledH      spatial extent in height
 * @param spatialScale spatial scale
 * @tparam T Numeric type. Only support float/double now
 */
class RoiPooling[T: ClassTag](val pooledW: Int, val pooledH: Int, val spatialScale: T)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  private val argmax: Tensor[T] = Tensor[T]
  private val gradInputTensor: Tensor[T] = Tensor[T]
  gradInput.insert(gradInputTensor)

  override def updateOutput(input: Table): Tensor[T] = {
    require(input.length() == 2,
      "there must have two tensors in the table," +
      s" number of tensors ${input.length()}")

    val data = input[Tensor[T]](1) // Input data to ROIPooling
    val rois = input[Tensor[T]](2) // Input label to ROIPooling

    require(rois.size().length > 1 && rois.size(2) == 5,
      "roi input shape should be (R, 5), " +
        s"input shape [${rois.size().length},${rois.size(2)}]")

    output.resize(rois.size(1), data.size(2), pooledH, pooledW)
      .fill(ev.fromType[Double](Double.MinValue))
    argmax.resizeAs(output).fill(ev.fromType(-1))
    val dataOffset = offset(0, 1, sizes = data.size())
    val argmaxOffset = offset(0, 1, sizes = argmax.size())
    val outputOffset = offset(0, 1, sizes = output.size())
    val roisOffset = offset(1, sizes = rois.size())
    var n = 0
    val dataSize = data.size()
    if (classTag[T] == classTag[Double]) {
      val inputData = data.storage().array().asInstanceOf[Array[Double]]
      val argmaxData = argmax.storage().array().asInstanceOf[Array[Double]]
      val outputData = output.storage().array().asInstanceOf[Array[Double]]
      val roisDouble = rois.asInstanceOf[Tensor[Double]]
      while (n < rois.size(1)) {
        poolOneRoiDouble(n, roisDouble(n + 1),
          inputData, dataSize, dataOffset,
          argmaxData, argmaxOffset,
          outputData, outputOffset,
          roisOffset, ev.toType[Double](spatialScale))
        n += 1
      }
    } else if (classTag[T] == classTag[Float]) {
      val inputData = data.storage().array().asInstanceOf[Array[Float]]
      val argmaxData = argmax.storage().array().asInstanceOf[Array[Float]]
      val outputData = output.storage().array().asInstanceOf[Array[Float]]
      val roisFloat = rois.asInstanceOf[Tensor[Float]]
      while (n < rois.size(1)) {
        poolOneRoiFloat(n, roisFloat(n + 1),
          inputData, dataSize, dataOffset,
          argmaxData, argmaxOffset,
          outputData, outputOffset,
          roisOffset, ev.toType[Float](spatialScale))
        n += 1
      }
    } else {
      throw new IllegalArgumentException("currently only Double and Float types are supported")
    }

    output
  }

  private def scaleRoiFloat(roi: Tensor[Float], ind: Int, spatialScale: Float): Int = {
    Math.round(roi.valueAt(ind) * spatialScale)
  }

  private def poolOneRoiFloat(n: Int, roi: Tensor[Float],
    inputData: Array[Float], dataSize: Array[Int], dataOffset: Int,
    argmaxData: Array[Float], argmaxOffset: Int,
    outputData: Array[Float], outputOffset: Int, roisOffset: Int,
    spatialScale: Float): Unit = {
    val roiBatchInd = roi.valueAt(1).toInt
    val roiStartW = scaleRoiFloat(roi, 2, spatialScale)
    val roiStartH = scaleRoiFloat(roi, 3, spatialScale)

    require(roiBatchInd >= 0 && dataSize(0) > roiBatchInd)

    val binSizeH = Math.max(scaleRoiFloat(roi, 5, spatialScale) - roiStartH + 1, 1f) / pooledH
    val binSizeW = Math.max(scaleRoiFloat(roi, 4, spatialScale) - roiStartW + 1, 1f) / pooledW
    var batchDataIndex = offset(roiBatchInd, sizes = dataSize)

    var c = 0
    while (c < dataSize(1)) {
      var ph = 0
      val outputDataIndex = outputOffset * (n * dataSize(1) + c)
      val argmaxIndex = argmaxOffset * (n * dataSize(1) + c)
      while (ph < pooledH) {
        var pw = 0
        while (pw < pooledW) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          val hstart = Math.min(Math.max(Math.floor(ph * binSizeH).toInt + roiStartH, 0),
            dataSize(2))
          val hend = Math.min(Math.max(Math.ceil((ph + 1) * binSizeH).toInt + roiStartH, 0),
            dataSize(2))
          val wstart = Math.min(Math.max(Math.floor(pw * binSizeW).toInt + roiStartW, 0),
            dataSize(3))
          val wend = Math.min(Math.max(Math.ceil((pw + 1) * binSizeW).toInt + roiStartW, 0),
            dataSize(3))

          val poolIndex = ph * pooledW + pw
          if ((hend <= hstart) || (wend <= wstart)) {
            outputData(outputDataIndex + poolIndex) = 0
            argmaxData(argmaxIndex + poolIndex) = -1
          } else {
            var h = hstart
            while (h < hend) {
              var w = wstart
              val hi = h * dataSize(3)
              while (w < wend) {
                val index = hi + w
                if (inputData(batchDataIndex + index) >
                  outputData(outputDataIndex + poolIndex)) {
                  outputData(outputDataIndex + poolIndex) = inputData(batchDataIndex + index)
                  argmaxData(argmaxIndex + poolIndex) = index
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

  private def scaleRoiDouble(roi: Tensor[Double], ind: Int, spatialScale: Double): Int = {
    Math.round(roi.valueAt(ind) * spatialScale).toInt
  }

  private def poolOneRoiDouble(n: Int, roi: Tensor[Double],
    inputData: Array[Double], dataSize: Array[Int], dataOffset: Int,
    argmaxData: Array[Double], argmaxOffset: Int,
    outputData: Array[Double], outputOffset: Int, roisOffset: Int,
    spatialScale: Double): Unit = {
    val roiBatchInd = roi.valueAt(1).toInt
    val roiStartW = scaleRoiDouble(roi, 2, spatialScale)
    val roiStartH = scaleRoiDouble(roi, 3, spatialScale)

    require(roiBatchInd >= 0 && dataSize(0) > roiBatchInd)

    val binSizeH = Math.max(scaleRoiDouble(roi, 5, spatialScale) - roiStartH + 1, 1f) / pooledH
    val binSizeW = Math.max(scaleRoiDouble(roi, 4, spatialScale) - roiStartW + 1, 1f) / pooledW
    var batchDataIndex = offset(roiBatchInd, sizes = dataSize)

    var c = 0
    while (c < dataSize(1)) {
      var ph = 0
      val outputDataIndex = outputOffset * (n * dataSize(1) + c)
      val argmaxIndex = argmaxOffset * (n * dataSize(1) + c)
      while (ph < pooledH) {
        var pw = 0
        while (pw < pooledW) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          val hstart = Math.min(Math.max(Math.floor(ph * binSizeH).toInt + roiStartH, 0),
            dataSize(2))
          val hend = Math.min(Math.max(Math.ceil((ph + 1) * binSizeH).toInt + roiStartH, 0),
            dataSize(2))
          val wstart = Math.min(Math.max(Math.floor(pw * binSizeW).toInt + roiStartW, 0),
            dataSize(3))
          val wend = Math.min(Math.max(Math.ceil((pw + 1) * binSizeW).toInt + roiStartW, 0),
            dataSize(3))

          val poolIndex = ph * pooledW + pw
          if ((hend <= hstart) || (wend <= wstart)) {
            outputData(outputDataIndex + poolIndex) = 0
            argmaxData(argmaxIndex + poolIndex) = -1
          } else {
            var h = hstart
            while (h < hend) {
              var w = wstart
              val hi = h * dataSize(3)
              while (w < wend) {
                val index = hi + w
                if (inputData(batchDataIndex + index) >
                  outputData(outputDataIndex + poolIndex)) {
                  outputData(outputDataIndex + poolIndex) = inputData(batchDataIndex + index)
                  argmaxData(argmaxIndex + poolIndex) = index
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


  /**
   * get the data offset given n, c, h, w
   * @param n     batch indice
   * @param c     channel indice
   * @param h     height indice
   * @param w     width indice
   * @param sizes tensor size
   * @return array offset
   */
  private def offset(n: Int, c: Int = 0, h: Int = 0, w: Int = 0, sizes: Array[Int]): Int = {
    require(sizes.length == 2 || sizes.length >= 4)
    if (sizes.length == 2) ((n * sizes(1) + c) + h) + w
    else ((n * sizes(1) + c) * sizes(2) + h) * sizes(3) + w
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val numRois = output.size(1)
    if (classTag[T] == classTag[Double]) {
      val data = input[Tensor[Double]](1)
      val roisData = input[Tensor[Double]](2).storage().array()
      val argmaxData = argmax.storage().array().asInstanceOf[Array[Double]]
      val gradInputData = gradInputTensor.resizeAs(data).zero()
        .storage().array().asInstanceOf[Array[Double]]
      val gradOutputData = gradOutput.storage().array().asInstanceOf[Array[Double]]
      roiPoolingBackwardDouble(roisData, numRois, data,
        argmaxData, gradInputData, gradOutputData)
    } else if (classTag[T] == classTag[Float]) {
      val data = input[Tensor[Float]](1)
      val roisData = input[Tensor[Float]](2).storage().array()
      val argmaxData = argmax.storage().array().asInstanceOf[Array[Float]]
      val gradInputData = gradInputTensor.resizeAs(data).zero()
        .storage().array().asInstanceOf[Array[Float]]
      val gradOutputData = gradOutput.storage().array().asInstanceOf[Array[Float]]
      roiPoolingBackwardFloat(roisData, numRois, data,
        argmaxData, gradInputData, gradOutputData)
    } else {
      throw new IllegalArgumentException("currently only Double and Float types are supported")
    }
    gradInput
  }

  private def roiPoolingBackwardFloat(roisData: Array[Float], numRois: Int, data: Tensor[Float],
    argmaxData: Array[Float], gradInputData: Array[Float], gradOutputData: Array[Float]): Unit = {
    var roiN = 0
    var c = 0
    var ph = 0
    var pw = 0
    // Accumulate gradient over all ROIs
    while (roiN < numRois) {
      val roiBatchInd = roisData(roiN * 5).toInt
      // Accumulate gradients over each bin in this ROI
      c = 0
      while (c < data.size(2)) {
        ph = 0
        while (ph < pooledH) {
          pw = 0
          while (pw < pooledW) {
            val outputOffset = ((roiN * data.size(2) + c) * pooledH + ph) * pooledW + pw
            val argmaxIndex = argmaxData(outputOffset)
            if (argmaxIndex >= 0) {
              val inputOffset = (roiBatchInd * data.size(2)
                + c) * data.size(3) * data.size(4) + argmaxIndex.toInt
              gradInputData(inputOffset) = gradInputData(inputOffset) + gradOutputData(outputOffset)
            }
            pw += 1
          }
          ph += 1
        }
        c += 1
      }
      roiN += 1
    }
  }

  private def roiPoolingBackwardDouble(roisData: Array[Double], numRois: Int, data: Tensor[Double],
    argmaxData: Array[Double], gradInputData: Array[Double],
    gradOutputData: Array[Double]): Unit = {
    var roiN = 0
    var c = 0
    var ph = 0
    var pw = 0
    // Accumulate gradient over all ROIs
    while (roiN < numRois) {
      val roiBatchInd = roisData(roiN * 5).toInt
      // Accumulate gradients over each bin in this ROI
      c = 0
      while (c < data.size(2)) {
        ph = 0
        while (ph < pooledH) {
          pw = 0
          while (pw < pooledW) {
            val outputOffset = ((roiN * data.size(2) + c) * pooledH + ph) * pooledW + pw
            val argmaxIndex = argmaxData(outputOffset)
            if (argmaxIndex >= 0) {
              val inputOffset = (roiBatchInd * data.size(2)
                + c) * data.size(3) * data.size(4) + argmaxIndex.toInt
              gradInputData(inputOffset) = gradInputData(inputOffset) + gradOutputData(outputOffset)
            }
            pw += 1
          }
          ph += 1
        }
        c += 1
      }
      roiN += 1
    }
  }

  override def toString: String = "nn.RoiPooling"

  override def clearState(): this.type = {
    super.clearState()
    argmax.set()
    gradInputTensor.set()
    this
  }
}

object RoiPooling {
  def apply[@specialized(Float, Double) T: ClassTag](
    pooled_w: Int, pooled_h: Int, spatial_scale: T)(implicit ev: TensorNumeric[T]): RoiPooling[T] =
    new RoiPooling[T](pooled_w, pooled_h, spatial_scale)
}
