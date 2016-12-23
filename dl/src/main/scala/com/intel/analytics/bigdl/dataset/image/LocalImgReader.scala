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

import java.awt.color.ColorSpace

import com.intel.analytics.bigdl.dataset.Transformer

import scala.collection.Iterator

object LocalImgReader {
  Class.forName("javax.imageio.ImageIO")
  Class.forName("java.awt.color.ICC_ColorSpace")
  Class.forName("sun.java2d.cmm.lcms.LCMS")
  ColorSpace.getInstance(ColorSpace.CS_sRGB).toRGB(Array[Float](0, 0, 0))

  def apply(scaleTo: Int = RGBImage.NO_SCALE, normalize: Float = 255f)
  : Transformer[LocalLabeledImagePath, LabeledRGBImage]
  = new LocalScaleImgReader(scaleTo, normalize)

  def apply(resizeW: Int, resizeH: Int, normalize: Float)
  : Transformer[LocalLabeledImagePath, LabeledRGBImage]
  = new LocalResizeImgReader(resizeW, resizeH, normalize)
}

class LocalScaleImgReader private[dataset](scaleTo: Int, normalize: Float)
  extends Transformer[LocalLabeledImagePath, LabeledRGBImage] {


  private val buffer = new LabeledRGBImage()

  override def apply(prev: Iterator[LocalLabeledImagePath]): Iterator[LabeledRGBImage] = {
    prev.map(data => {
      val imgData = RGBImage.readImage(data.path, scaleTo)
      val label = data.label
      buffer.copy(imgData, normalize).setLabel(label)
    })
  }
}

class LocalResizeImgReader private[dataset](resizeW: Int, resizeH: Int, normalize: Float)
  extends Transformer[LocalLabeledImagePath, LabeledRGBImage] {


  private val buffer = new LabeledRGBImage()

  override def apply(prev: Iterator[LocalLabeledImagePath]): Iterator[LabeledRGBImage] = {
    prev.map(data => {
      val imgData = RGBImage.readImage(data.path, resizeW, resizeH)
      val label = data.label
      buffer.copy(imgData, normalize).setLabel(label)
    })
  }
}
