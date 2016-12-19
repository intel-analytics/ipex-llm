/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.dataset.image

import com.intel.analytics.bigdl.dataset.Transformer

import scala.collection.Iterator

object RGBImgRdmCropper {
  def apply(cropWidth: Int, cropHeight: Int, padding: Int): RGBImgRdmCropper =
    new RGBImgRdmCropper(cropHeight, cropWidth, padding)
}

class RGBImgRdmCropper(cropHeight: Int, cropWidth: Int, padding: Int)
  extends Transformer[LabeledRGBImage, LabeledRGBImage] {
  import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

  private val buffer = new LabeledRGBImage(cropWidth, cropHeight)

  override def apply(prev: Iterator[LabeledRGBImage]): Iterator[LabeledRGBImage] = {
    prev.map(img => {
      val curImg = padding > 0 match {
        case true => {
          val widthTmp = img.width()
          val heightTmp = img.height()
          val sourceTmp = img.content
          val padWidth = widthTmp + 2 * padding
          val padHeight = heightTmp + 2 * padding
          val temp = new LabeledRGBImage(padWidth, padHeight)
          val tempBuffer = temp.content
          val startIndex = (padding + 1 + (padding + 1) * padWidth) * 3
          val frameLength = widthTmp * heightTmp
          var i = 0
          while (i < frameLength) {
            tempBuffer(startIndex + ((i / widthTmp) * padWidth + (i % widthTmp)) * 3 + 2) = sourceTmp(i * 3 + 2)
            tempBuffer(startIndex + ((i / widthTmp) * padWidth + (i % widthTmp)) * 3 + 1) = sourceTmp(i * 3 + 1)
            tempBuffer(startIndex + ((i / widthTmp) * padWidth + (i % widthTmp)) * 3) = sourceTmp(i * 3)
            i += 1
          }
          temp.setLabel(img.label())
          temp
        }
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