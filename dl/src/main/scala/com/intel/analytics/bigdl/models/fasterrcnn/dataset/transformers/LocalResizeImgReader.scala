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

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.models.fasterrcnn.dataset.{BGRImageOD, ImageWithRoi, Roidb, Target}

object LocalResizeImgReader {
  def apply(scales: Array[Int], scaleMultipleOf: Int, resizeRois: Boolean): LocalResizeImgReader =
    new LocalResizeImgReader(scales, scaleMultipleOf, resizeRois)
}

class LocalResizeImgReader(scales: Array[Int], scaleMultipleOf: Int, resizeRois: Boolean)
  extends Transformer[Roidb, ImageWithRoi] {

  private val buffer = new ImageWithRoi()

  override def apply(prev: Iterator[Roidb]): Iterator[ImageWithRoi] = {
    prev.map(data => {
      transform(data)
    })
  }

  def transform(roidb: Roidb, useBuffer: Boolean = true): ImageWithRoi = {
    val imageWithRoi = if (useBuffer) buffer else new ImageWithRoi()
    val imgData = BGRImageOD.readImage(Paths.get(roidb.imagePath), scales, scaleMultipleOf,
      imageWithRoi.imInfo)
    imageWithRoi.copy(imgData)
    imageWithRoi.path = roidb.imagePath
    if (resizeRois) {
      imageWithRoi.target = roidb.resizeGtRois(imageWithRoi)
    } else {
      imageWithRoi.target = Some(Target(roidb.gtClasses, roidb.gtBoxes))
    }
    imageWithRoi
  }
}
