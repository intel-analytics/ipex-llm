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

package com.intel.analytics.bigdl.dataset.image

import com.intel.analytics.bigdl.dataset.Transformer

import scala.collection.Iterator

object BGRImgRdmCropper {
  def apply(cropWidth: Int, cropHeight: Int, padding: Int): BGRImgRdmCropper =
    new BGRImgRdmCropper(cropHeight, cropWidth, padding)
}

/**
 * Random crop a specified area from the Image. The result is also an image
 * @param cropHeight crop area height
 * @param cropWidth crop area width
 * @param padding padding of the image area to be cropped
 */
class BGRImgRdmCropper(cropHeight: Int, cropWidth: Int, padding: Int)
  extends Transformer[LabeledBGRImage, LabeledBGRImage] {
  import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

  private val buffer = new LabeledBGRImage(cropWidth, cropHeight)

  override def apply(prev: Iterator[LabeledBGRImage]): Iterator[LabeledBGRImage] = {
    prev.map(img => {
      val curImg = padding > 0 match {
        case true =>
          val widthTmp = img.width()
          val heightTmp = img.height()
          val sourceTmp = img.content
          val padWidth = widthTmp + 2 * padding
          val padHeight = heightTmp + 2 * padding
          val temp = new LabeledBGRImage(padWidth, padHeight)
          val tempBuffer = temp.content
          val startIndex = (padding + padding * padWidth) * 3
          val frameLength = widthTmp * heightTmp
          var i = 0
          while (i < frameLength) {
            tempBuffer(startIndex +
              ((i / widthTmp) * padWidth + (i % widthTmp)) * 3 + 2) = sourceTmp(i * 3 + 2)
            tempBuffer(startIndex +
              ((i / widthTmp) * padWidth + (i % widthTmp)) * 3 + 1) = sourceTmp(i * 3 + 1)
            tempBuffer(startIndex +
              ((i / widthTmp) * padWidth + (i % widthTmp)) * 3) = sourceTmp(i * 3)
            i += 1
          }
          temp.setLabel(img.label())
          temp
        case _ => img
      }

      val width = curImg.width()
      val height = curImg.height()
      val source = curImg.content

      val startW = RNG.uniform(0, width - cropWidth).toInt
      val startH = RNG.uniform(0, height - cropHeight).toInt
      val startIndex = (startW + startH * width) * 3
      val frameLength = cropWidth * cropHeight

      val target = buffer.content
      var i = 0
      while (i < frameLength) {
        target(i * 3 + 2) =
          source(startIndex + ((i / cropWidth) * width + (i % cropWidth)) * 3 + 2)
        target(i * 3 + 1) =
          source(startIndex + ((i / cropWidth) * width + (i % cropWidth)) * 3 + 1)
        target(i * 3) =
          source(startIndex + ((i / cropWidth) * width + (i % cropWidth)) * 3)
        i += 1
      }
      buffer.setLabel(curImg.label())
    })
  }
}
