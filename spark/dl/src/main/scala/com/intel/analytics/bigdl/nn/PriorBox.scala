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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.{NumericDouble, NumericFloat}
import com.intel.analytics.bigdl.utils.{Shape, SingleShape}

import scala.collection.mutable.ArrayBuffer
import scala.reflect._

/**
 * Generate the prior boxes of designated sizes and aspect ratios across
 * all dimensions (H * W)
 * Intended for use with MultiBox detection method to generate prior
 *
 * @param minSizes minimum box size in pixels. can be multiple. required!
 * @param maxSizes maximum box size in pixels. can be ignored or same as the
 * # of min_size.
 * @param _aspectRatios optional aspect ratios of the boxes. can be multiple
 * @param isFlip optional bool, default true. if set, flip the aspect ratio.
 * @param isClip whether to clip the prior's coordidate such that it is within [0, 1]
 * @tparam T Numeric type. Only support float/double now
 */
@SerialVersionUID(7934178172129260471L)
class PriorBox[T: ClassTag](minSizes: Array[Float], maxSizes: Array[Float] = null,
  _aspectRatios: Array[Float] = null, isFlip: Boolean = true, isClip: Boolean = false,
  var variances: Array[Float] = null, offset: Float = 0.5f,
  var imgH: Int = 0, var imgW: Int = 0, imgSize: Int = 0,
  var stepH: Float = 0, var stepW: Float = 0, step: Float = 0)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Tensor[T], T] {

  private var aspectRatios: ArrayBuffer[Float] = _
  private var numPriors = 0

  init()

  private def init(): Unit = {
    require(minSizes != null && minSizes.length > 0, "must provide minSize")
    if (aspectRatios == null) aspectRatios = new ArrayBuffer[Float]()
    else aspectRatios.clear()
    aspectRatios.append(1)
    var i = 0
    if (_aspectRatios != null) {
      while (i < _aspectRatios.length) {
        val ar = _aspectRatios(i)
        if (!checkExist(aspectRatios, ar)) {
          aspectRatios.append(ar)
        }
        if (isFlip) {
          aspectRatios.append(1 / ar)
        }
        i += 1
      }
    }

    numPriors = aspectRatios.length * minSizes.length
    if (maxSizes != null && maxSizes.length > 0) {
      require(minSizes.length == maxSizes.length)
      i = 0
      while (i < maxSizes.length) {
        require(maxSizes(i) > minSizes(i))
        numPriors += 1
        i += 1
      }
    }

    if (variances == null) {
      variances = Array[Float](0.1f)
    } else if (variances.length > 1) {
      require(variances.length == 4, "Must and only provide 4 variance.")
    }

    if (imgH != 0 && imgW != 0) {
      require(imgW > 0 && imgH > 0)
    } else if (imgSize != 0) {
      require(imgSize > 0)
      imgH = imgSize
      imgW = imgSize
    }

    if (stepH != 0 && stepW != 0) {
      require(stepW > 0 && stepH > 0)
    } else if (step != 0) {
      require(step > 0)
      stepH = step
      stepW = step
    }
  }

  private def checkExist(aspectRatios: ArrayBuffer[Float], ar: Float): Boolean = {
    var j = 0
    while (j < aspectRatios.length) {
      if (Math.abs(ar - aspectRatios(j)) < 1e-6) {
        return true
      }
      j += 1
    }
    false
  }

  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   *
   * @param input
   * @return
   */
  override def updateOutput(input: Activity): Tensor[T] = {
    require(imgW > 0 && imgH > 0, "imgW and imgH must > 0")
    val feature = if (input.isTensor) input.toTensor[Float] else input.toTable[Tensor[Float]](1)
    val layerW = feature.size(4)
    val layerH = feature.size(3)
    if (stepW == 0 || stepH == 0) {
      stepW = imgW / layerW.toFloat
      stepH = imgH / layerH.toFloat
    }
    val dim = layerH * layerW * numPriors * 4
    if (output.nElement() == 2 * dim && output.dim() == 3 &&
      output.size(1) == 1 && output.size(2) == 2 && output.size(3) == dim) {
      return output
    }
    // Since all images in a batch has same height and width, we only need to
    // generate one set of priors which can be shared across all images.
    // 2 channels. First channel stores the mean of each prior coordinate.
    // Second channel stores the variance of each prior coordinate.
    output.resize(1, 2, dim)
    val offset = output.storageOffset() - 1
    if (classTag[T] == classTag[Float]) {
      val outputData = output.storage().array().asInstanceOf[Array[Float]]
      computPriorBoxFloat(layerW, layerH, imgW, imgH, dim, outputData, offset)
    } else if (classTag[T] == classTag[Double]) {
      val outputData = output.storage().array().asInstanceOf[Array[Double]]
      computPriorBoxDouble(layerW, layerH, imgW, imgH, dim, outputData, offset)
    }
    output
  }

  def computPriorBoxFloat(layerW: Int, layerH: Int, imgWidth: Float, imgHeight: Float,
    dim: Int, outputData: Array[Float], outputOffset: Int): Unit = {
    var idx = outputOffset
    var h = 0
    while (h < layerH) {
      var w = 0
      while (w < layerW) {
        val centerX = (w + offset) * stepW
        val centerY = (h + offset) * stepH
        var halfBoxW = 0f
        var halfBoxH = 0f
        var s = 0
        while (s < minSizes.length) {
          val minSize = minSizes(s).toInt
          halfBoxW = minSize / 2.0f
          halfBoxH = minSize / 2.0f
          outputData(idx) = (centerX - halfBoxW) / imgWidth // xmin
          outputData(idx + 1) = (centerY - halfBoxH) / imgHeight // ymin
          outputData(idx + 2) = (centerX + halfBoxW) / imgWidth // xmax
          outputData(idx + 3) = (centerY + halfBoxH) / imgHeight // ymax
          idx += 4
          if (maxSizes != null && maxSizes.length > 0) {
            val maxSize = maxSizes(s).toInt
            // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
            halfBoxW = Math.sqrt(minSize * maxSize).toFloat / 2
            halfBoxH = halfBoxW
            outputData(idx) = (centerX - halfBoxW) / imgWidth // xmin
            outputData(idx + 1) = (centerY - halfBoxH) / imgHeight // ymin
            outputData(idx + 2) = (centerX + halfBoxW) / imgWidth // xmax
            outputData(idx + 3) = (centerY + halfBoxH) / imgHeight // ymax
            idx += 4
          }

          var r = 0
          // rest of priors
          while (r < aspectRatios.length) {
            val ar = aspectRatios(r)
            if (Math.abs(ar - 1) >= 1e-6) {
              val v = Math.sqrt(ar).toFloat
              halfBoxW = minSize * v / 2
              halfBoxH = minSize / v / 2
              outputData(idx) = (centerX - halfBoxW) / imgWidth // xmin
              outputData(idx + 1) = (centerY - halfBoxH) / imgHeight // ymin
              outputData(idx + 2) = (centerX + halfBoxW) / imgWidth // xmax
              outputData(idx + 3) = (centerY + halfBoxH) / imgHeight // ymax
              idx += 4
            }
            r += 1
          }
          s += 1
        }
        w += 1
      }
      h += 1
    }
    // clip the prior's coordidate such that it is within [0, 1]
    if (isClip) {
      var d = outputOffset
      while (d < dim) {
        outputData(d) = Math.min(Math.max(outputData(d), 0), 1)
        d += 1
      }
    }
    // set the variance.
    // var outputDataOffset = output.storageOffset() - 1 + offset(0, 1, sizes = output.size())
    if (variances.length == 1) {
      NumericFloat.fill(outputData, idx, output.nElement(), variances(0))
    } else {
      var d = 0
      while (d < dim) {
        Array.copy(variances, 0, outputData, idx, 4)
        idx += 4
        d += 4
      }
    }
  }

  def computPriorBoxDouble(layerW: Int, layerH: Int, imgWidth: Float, imgHeight: Float,
    dim: Int, outputData: Array[Double], outputOffset: Int): Unit = {
    var idx = outputOffset
    var h = 0
    while (h < layerH) {
      var w = 0
      while (w < layerW) {
        val centerX = (w + offset) * stepW
        val centerY = (h + offset) * stepH
        var halfBoxW = 0f
        var halfBoxH = 0f
        var s = 0
        while (s < minSizes.length) {
          val minSize = minSizes(s)
          halfBoxW = minSize / 2
          halfBoxH = minSize / 2
          outputData(idx) = (centerX - halfBoxW) / imgWidth // xmin
          outputData(idx + 1) = (centerY - halfBoxH) / imgHeight // ymin
          outputData(idx + 2) = (centerX + halfBoxW) / imgWidth // xmax
          outputData(idx + 3) = (centerY + halfBoxH) / imgHeight // ymax
          idx += 4
          if (maxSizes != null && maxSizes.length > 0) {
            val maxSize = maxSizes(s)
            // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
            halfBoxW = Math.sqrt(minSize * maxSize).toFloat / 2
            halfBoxH = halfBoxW
            outputData(idx) = (centerX - halfBoxW) / imgWidth // xmin
            outputData(idx + 1) = (centerY - halfBoxH) / imgHeight // ymin
            outputData(idx + 2) = (centerX + halfBoxW) / imgWidth // xmax
            outputData(idx + 3) = (centerY + halfBoxH) / imgHeight // ymax
            idx += 4
          }

          var r = 0
          // rest of priors
          while (r < aspectRatios.length) {
            val ar = aspectRatios(r)
            if (Math.abs(ar - 1) >= 1e-6) {
              val v = Math.sqrt(ar).toFloat
              halfBoxW = minSize * v / 2
              halfBoxH = minSize / v / 2
              outputData(idx) = (centerX - halfBoxW) / imgWidth // xmin
              outputData(idx + 1) = (centerY - halfBoxH) / imgHeight // ymin
              outputData(idx + 2) = (centerX + halfBoxW) / imgWidth // xmax
              outputData(idx + 3) = (centerY + halfBoxH) / imgHeight // ymax
              idx += 4
            }
            r += 1
          }
          s += 1
        }
        w += 1
      }
      h += 1
    }
    // clip the prior's coordidate such that it is within [0, 1]
    if (isClip) {
      var d = outputOffset
      while (d < dim) {
        outputData(d) = Math.min(Math.max(outputData(d), 0), 1)
        d += 1
      }
    }
    // set the variance.
    // var outputDataOffset = output.storageOffset() - 1 + offset(0, 1, sizes = output.size())
    if (variances.length == 1) {
      NumericDouble.fill(outputData, idx, output.nElement(), variances(0))
    } else {
      var d = 0
      while (d < dim) {
        Array.copy(variances, 0, outputData, idx, 4)
        idx += 4
        d += 4
      }
    }
  }

  override def updateGradInput(input: Activity, gradOutput: Tensor[T]): Activity = {
    gradInput = null
    gradInput
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val feature = if (inputShape.isInstanceOf[SingleShape]) {
      inputShape.toSingle().toArray
    } else {
      inputShape.toMulti().toArray.apply(0).toSingle().toArray
    }
    val layerW = feature(3)
    val layerH = feature(2)
    if (stepW == 0 || stepH == 0) {
      stepW = imgW / layerW.toFloat
      stepH = imgH / layerH.toFloat
    }
    val dim = layerH * layerW * numPriors * 4
    val outputSize = if (output.nElement() == 2 * dim && output.dim() == 3 &&
      output.size(1) == 1 && output.size(2) == 2 && output.size(3) == dim) {
      output.size()
    } else Array(1, 2, dim)

    Shape(outputSize)
  }
}

object PriorBox {
  def apply[@specialized(Float, Double) T: ClassTag](minSizes: Array[Float],
    maxSizes: Array[Float] = null,
    _aspectRatios: Array[Float] = null, isFlip: Boolean = true, isClip: Boolean = false,
    variances: Array[Float] = null, offset: Float = 0.5f,
    imgH: Int = 0, imgW: Int = 0, imgSize: Int = 0,
    stepH: Float = 0, stepW: Float = 0, step: Float = 0)
    (implicit ev: TensorNumeric[T]): PriorBox[T] =
    new PriorBox[T](minSizes, maxSizes, _aspectRatios, isFlip, isClip, variances, offset, imgH,
      imgW, imgSize, stepH, stepW, step)
}
