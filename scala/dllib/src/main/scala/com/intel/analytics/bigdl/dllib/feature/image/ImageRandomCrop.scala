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

import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.augmentation

/**
 * Random crop a `cropWidth` x `cropHeight` patch from an image.
 * The patch size should be less than the image size.
 *
 * @param cropWidth width after crop
 * @param cropHeight height after crop
 * @param isClip whether to clip the roi to image boundaries
 */
class ImageRandomCrop(cropWidth: Int, cropHeight: Int, isClip: Boolean = true)
  extends ImageProcessing {

  private val internalCrop = new augmentation.RandomCrop(cropWidth, cropHeight, isClip)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalCrop.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalCrop.transformMat(feature)
  }
}

object ImageRandomCrop {
  def apply(cropWidth: Int, cropHeight: Int, isClip: Boolean = true): ImageRandomCrop =
    new ImageRandomCrop(cropWidth, cropHeight, isClip)
}
