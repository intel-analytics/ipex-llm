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

package com.intel.analytics.bigdl.models.fasterrcnn.dataset.transformers

import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.image.BGRImage
import com.intel.analytics.bigdl.models.fasterrcnn.dataset.{BGRImageOD, ByteImageWithRoi, ImageWithRoi, Target}

object ImageResizer {
  def apply(scales: Array[Int], scaleMultipleOf: Int, resizeRois: Boolean): ImageResizer =
    new ImageResizer(scales, scaleMultipleOf, resizeRois)
}
class ImageResizer(scales: Array[Int], scaleMultipleOf: Int, resizeRois: Boolean)
  extends Transformer[ByteImageWithRoi, ImageWithRoi] {

  val imageWithRoi = new ImageWithRoi()

  override def apply(prev: Iterator[ByteImageWithRoi]): Iterator[ImageWithRoi] = {
    prev.map(x => {
      // convert byte array back to BufferedImage
      val img = ImageIO.read(new ByteArrayInputStream(x.data, 0, x.dataLength))
      BGRImageOD.getWidthHeightAfterRatioScale(img, scales, scaleMultipleOf, imageWithRoi.imInfo)
      val scaledImage = BGRImage.resizeImage(img, imageWithRoi.imInfo.valueAt(2).toInt,
        imageWithRoi.imInfo.valueAt(1).toInt)
      imageWithRoi.copy(scaledImage)
      imageWithRoi.path = x.path
      if (resizeRois) {
        imageWithRoi.target = x.resizeGtRois(imageWithRoi)
      } else {
        imageWithRoi.target = Some(Target(x.gtClasses, x.gtBoxes))
      }
      imageWithRoi
    })
  }

}
