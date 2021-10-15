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
package com.intel.analytics.bigdl.dllib.feature.image

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature

/**
 * Resize loading image
 * @param resizeH height after resize
 * @param resizeW width after resize
 */
class BufferedImageResize(resizeH: Int, resizeW: Int) extends ImageProcessing {

  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    prev.map { imf =>
      val img = ImageIO.read(new ByteArrayInputStream(imf.bytes()))
      if ((resizeH != img.getHeight) || (resizeW != img.getWidth)) {
        val scaledImage =
          img.getScaledInstance(resizeW, resizeH, java.awt.Image.SCALE_SMOOTH)

        val imageBuff: BufferedImage =
                  new BufferedImage(resizeW, resizeH, img.getType)
        imageBuff.getGraphics().drawImage(scaledImage, 0, 0, null)

        val output = new ByteArrayOutputStream()
        val uri = imf[String](ImageFeature.uri)
        ImageIO.write(imageBuff, uri.substring(uri.lastIndexOf(".") + 1), output)
        imf(ImageFeature.bytes) = output.toByteArray
      }
      imf
    }
  }
}

object BufferedImageResize {
  def apply(resizeH: Int, resizeW: Int): BufferedImageResize =
    new BufferedImageResize(resizeH, resizeW)
}
