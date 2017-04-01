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

import java.awt.color.ColorSpace

import com.intel.analytics.bigdl.dataset.Transformer

import scala.collection.Iterator

object LocalImgReader {
  Class.forName("javax.imageio.ImageIO")
  Class.forName("java.awt.color.ICC_ColorSpace")
  // Class.forName("sun.java2d.cmm.lcms.LCMS")
  ColorSpace.getInstance(ColorSpace.CS_sRGB).toRGB(Array[Float](0, 0, 0))

  def apply(scaleTo: Int = BGRImage.NO_SCALE, normalize: Float = 255f)
  : Transformer[LocalLabeledImagePath, LabeledBGRImage]
  = new LocalScaleImgReader(scaleTo, normalize)

  def apply(resizeW: Int, resizeH: Int, normalize: Float)
  : Transformer[LocalLabeledImagePath, LabeledBGRImage]
  = new LocalResizeImgReader(resizeW, resizeH, normalize)
}

/**
 * Read BGR images from local given paths. After read the image, it will resize the shorted
 * edge to the given scale to value and resize the other edge properly. It will also divide
 * the pixel value by the given normalize value.
 * @param scaleTo
 * @param normalize
 */
class LocalScaleImgReader private[dataset](scaleTo: Int, normalize: Float)
  extends Transformer[LocalLabeledImagePath, LabeledBGRImage] {


  private val buffer = new LabeledBGRImage()

  override def apply(prev: Iterator[LocalLabeledImagePath]): Iterator[LabeledBGRImage] = {
    prev.map(data => {
      val imgData = BGRImage.readImage(data.path, scaleTo)
      val label = data.label
      buffer.copy(imgData, normalize).setLabel(label)
    })
  }
}

/**
 * Read BGR images from local given paths. After read the image, it will resize the images
 * to the given width and height.
 * Besides, it will also divide the pixel value by the given normalize value.
 * @param resizeW the given width to resize
 * @param resizeH the given hight to resize
 * @param normalize the value to normalize
 */
class LocalResizeImgReader private[dataset](resizeW: Int, resizeH: Int, normalize: Float)
  extends Transformer[LocalLabeledImagePath, LabeledBGRImage] {


  private val buffer = new LabeledBGRImage()

  override def apply(prev: Iterator[LocalLabeledImagePath]): Iterator[LabeledBGRImage] = {
    prev.map(data => {
      val imgData = BGRImage.readImage(data.path, resizeW, resizeH)
      val label = data.label
      buffer.copy(imgData, normalize).setLabel(label)
    })
  }
}
