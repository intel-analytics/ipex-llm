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
package com.intel.analytics.bigdl.transform.vision.image.augmentation

import com.intel.analytics.bigdl.dataset.image.{CropCenter, CropRandom, CropperMethod}
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.utils.RandomGenerator
import org.opencv.core.CvType

object RandomCropper {
  def apply(cropWidth: Int, cropHeight: Int,
            mirror: Boolean, cropperMethod: CropperMethod = CropRandom,
            channels: Int = 3): RandomCropper =
    new RandomCropper(cropHeight, cropWidth, mirror, cropperMethod, channels)
}

/**
 * Random cropper on uniform distribution with fixed height & width
 * @param cropWidth  width cropped to
 * @param cropHeight height cropped to
 * @param mirror   whether mirror
 * @param cropperMethod crop method
 */
class RandomCropper(cropWidth: Int, cropHeight: Int,
                      mirror: Boolean, cropperMethod: CropperMethod = CropRandom,
                      channels: Int = 3)
  extends FeatureTransformer {

  import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

  val buffer = new Array[Float](cropWidth * cropHeight * channels)

  override protected def transformMat(feature: ImageFeature): Unit = {
    val openCVMat = feature.opencvMat()

    if (openCVMat.`type`() != CvType.CV_32FC3) {
      openCVMat.convertTo(openCVMat, CvType.CV_32FC3)
    }

    val tmp = new Array[Float](openCVMat.height() * openCVMat.width() * openCVMat.channels())

    openCVMat.get(0, 0, tmp)

    val height = openCVMat.size().height.toInt
    val width = openCVMat.size().width.toInt

    val (startH, startW) = cropperMethod match {
      case CropRandom =>
        val indexH = math.ceil(RNG.uniform(0, height - cropHeight)).toInt
        val indexW = math.ceil(RNG.uniform(0, width - cropWidth)).toInt
        (indexH, indexW)
      case CropCenter =>
        ((height - cropHeight) / 2, (width - cropWidth) / 2)
    }
    val do_mirror = mirror && (RandomGenerator.RNG.uniform(0, 2).toInt != 0)
    val input = feature.toTensor(ImageFeature.imageTensor)
    cropper(input.storage().array(), buffer,
      Array(height, width), Array(cropHeight, cropWidth), startH, startW, do_mirror)
    openCVMat.put(0, 0, buffer)
    val mate = OpenCVMat.fromFloats(buffer, cropHeight, cropWidth, 3)
    if (openCVMat != null) {
      openCVMat.release()
    }
    if (mate.`type`() != CvType.CV_32FC3) {
      mate.convertTo(mate, CvType.CV_32FC3)
    }
    mate.put(0, 0, buffer)
    feature(ImageFeature.mat) = mate
  }

  private def cropper(source: Array[Float], target: Array[Float], srcSize: Array[Int],
              tarSize: Array[Int], startH: Int, startW: Int, mirror: Boolean = false): Unit = {
    val height = srcSize(0)
    val width = srcSize(1)
    val cropHeight = tarSize(0)
    val cropWidth = tarSize(1)

    val startIndex = startW + startH * width
    val frameLength = cropWidth * cropHeight
    var i = 0
    var c = 0
    val channels = 3
    while (c < channels) {
      i = 0
      while (i < frameLength) {
        val th = i / cropWidth
        val tw = i % cropWidth
        val data_index = (c * height + startH + th) * width + startW + tw
        val top_index = if (mirror) {
          ((th + 1) * cropWidth - tw - 1) * 3 + c
        } else {
          i * 3 + c
        }
        val data = source(data_index)
        target(top_index) = data
        i += 1
      }
      c += 1
    }
  }
}
