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

package com.intel.analytics.bigdl.dataset.roiimage

import java.io.ByteArrayOutputStream
import java.nio.file.Paths
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.image.BGRImage

/**
 * load local image and target if exists
 */
class LocalByteImageReader extends Transformer[RoiImagePath, RoiByteImage] {
  override def apply(prev: Iterator[RoiImagePath]): Iterator[RoiByteImage] = {
    prev.map(data => {
      transform(data)
    })
  }

  def transform(roiImagePath: RoiImagePath, useBuffer: Boolean = true): RoiByteImage = {
    val originalImage = BGRImage.readRawImage(Paths.get(roiImagePath.imagePath))
    // convert BufferedImage to byte array
    val baos = new ByteArrayOutputStream()
    // use jpg will lost some information
    ImageIO.write(originalImage, "png", baos)
    baos.flush()
    val imageInByte = baos.toByteArray
    baos.close()
    RoiByteImage(imageInByte, imageInByte.length, roiImagePath.target, roiImagePath.imagePath)
  }
}

object LocalByteImageReader {
  def apply(): LocalByteImageReader = new LocalByteImageReader()
}
