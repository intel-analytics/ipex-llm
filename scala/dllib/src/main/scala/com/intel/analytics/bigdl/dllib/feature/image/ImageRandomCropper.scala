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

import com.intel.analytics.bigdl.dllib.feature.dataset.image.{CropRandom, CropperMethod}
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.augmentation.RandomCropper

/**
 * Random cropper on uniform distribution with fixed height and width.
 *
 * @param cropWidth Integer. Width to be cropped to.
 * @param cropHeight Integer. Height to be cropped to.
 * @param mirror Boolean. Whether to do mirror.
 * @param cropperMethod An instance of [[CropperMethod]]. Default is [[CropRandom]].
 */
class ImageRandomCropper(cropWidth: Int, cropHeight: Int,
                         mirror: Boolean, cropperMethod: CropperMethod = CropRandom,
                         channels: Int = 3) extends ImageProcessing {
  private val internalTransformer = new InternalRandomCropper(cropWidth, cropHeight,
    mirror, cropperMethod, channels)

  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalTransformer.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalTransformer.transformMat(feature)
  }
}

object ImageRandomCropper {
  def apply(cropWidth: Int, cropHeight: Int,
            mirror: Boolean, cropperMethod: CropperMethod = CropRandom,
            channels: Int = 3): ImageRandomCropper = {
    new ImageRandomCropper(cropWidth, cropHeight, mirror, cropperMethod, channels)
  }
}

// transformMat in BigDL RandomCropper is protected and can't be directly accessed.
// Thus add an InternalRandomCropper here to override transformMat and make it accessible.
private[image] class InternalRandomCropper(cropWidth: Int, cropHeight: Int,
                            mirror: Boolean, cropperMethod: CropperMethod = CropRandom,
                            channels: Int = 3)
  extends RandomCropper(cropWidth, cropHeight, mirror, cropperMethod, channels) {

  override def transformMat(feature: ImageFeature): Unit = {
    super.transformMat(feature)
  }
}
