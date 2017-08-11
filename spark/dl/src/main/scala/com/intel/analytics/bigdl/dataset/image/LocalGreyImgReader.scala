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

object LocalGreyImgReader {
  Class.forName("javax.imageio.ImageIO")

  /**
   * Create grey image reader transformer to resize the shorted edge to the given scale to
   * value and resize the other edge properly. Also divide the pixel value by
   * the given normalize value
   * @param scaleTo scale to value
   * @param normalize the value to normalize
   * @return grey image reader transformer
   */
  def apply(scaleTo: Int = Image.NO_SCALE, normalize: Float = 255f, hasName: Boolean = false)
  : Transformer[LocalLabeledImagePath, LabeledGreyImage] = {
      new LocalScaleGreyImgReader(scaleTo, normalize, hasName)
  }


  /**
   * Create grey image reader transformer to resize the images to the given width and height.
   * And also divide the pixel value by the given normalize value.
   * @param resizeW the given width to resize
   * @param resizeH the given hight to resize
   * @param normalize the value to normalize
   * @return grey image reader transformer
   */
  def apply(resizeW: Int, resizeH: Int, normalize: Float, hasName: Boolean)
  : Transformer[LocalLabeledImagePath, LabeledGreyImage]
  = new LocalResizeGreyImgReader(resizeW, resizeH, normalize, hasName)
}

/**
 * Read Grey images from local given paths. After read the image, it will resize the shorted
 * edge to the given scale to value and resize the other edge properly. It will also divide
 * the pixel value by the given normalize value.
 * @param scaleTo the given scale to value
 * @param normalize the value to normalize
 * @param hasName whether the image contains name
  */
class LocalScaleGreyImgReader private[dataset](scaleTo: Int, normalize: Float, hasName: Boolean)
  extends Transformer[LocalLabeledImagePath, LabeledGreyImage] {


  private val buffer = new LabeledGreyImage()

  override def apply(prev: Iterator[LocalLabeledImagePath]): Iterator[LabeledGreyImage] = {
    prev.map(data => {
      val imgData = GreyImage.readImage(data.path, scaleTo)
      val label = data.label
      if (hasName) {
        buffer.copy(imgData, normalize).setLabel(label).setName(data.path.getFileName.toString)
      } else {
        buffer.copy(imgData, normalize).setLabel(label)
      }
    })
  }
}

/**
 * Read Grey images from local given paths. After read the image, it will resize the images
 * to the given width and height.
 * Besides, it will also divide the pixel value by the given normalize value.
 * @param resizeW the given width to resize
 * @param resizeH the given height to resize
 * @param normalize the value to normalize
 * @param hasName whether the image contains name
 */
class LocalResizeGreyImgReader private[dataset](resizeW: Int,
                                                resizeH: Int, normalize: Float, hasName: Boolean)
  extends Transformer[LocalLabeledImagePath, LabeledGreyImage] {


  private val buffer = new LabeledGreyImage()

  override def apply(prev: Iterator[LocalLabeledImagePath]): Iterator[LabeledGreyImage] = {
    prev.map(data => {
      val imgData = GreyImage.readImage(data.path, resizeW, resizeH)
      val label = data.label
      if (hasName) {
        buffer.copy(imgData, normalize).setLabel(label).setName(data.path.getFileName.toString)
      } else {
        buffer.copy(imgData, normalize).setLabel(label)
      }
    })
  }
}
