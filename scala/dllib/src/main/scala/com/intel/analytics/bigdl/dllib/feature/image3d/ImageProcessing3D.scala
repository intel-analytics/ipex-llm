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
package com.intel.analytics.bigdl.dllib.feature.image3d

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.FeatureTransformer._
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.dllib.feature.image.{ImageProcessing, ImageSet}
import org.apache.logging.log4j.LogManager

private[bigdl] abstract class ImageProcessing3D extends ImageProcessing {

  /**
   * if true, catch the exception of the transformer to avoid crashing.
   * if false, interrupt the transformer when error happens
   */
  private var ignoreImageException: Boolean = false

  /**
   * catch the exception of the transformer to avoid crashing.
   */
  override def enableIgnoreException(): this.type = {
    ignoreImageException = true
    this
  }


  protected def transformTensor(tensor: Tensor[Float]): Tensor[Float] = {
    tensor
  }

  /**
   * transform image feature
   *
   * @param feature ImageFeature
   * @return ImageFeature
   */
  override def transform(feature: ImageFeature): ImageFeature = {
    if (!feature.isInstanceOf[ImageFeature3D]) return feature
    else {
      transform(feature.asInstanceOf[ImageFeature3D])
    }
  }

  /**
   * transform 3D image feature
   *
   * @param feature ImageFeature3D
   * @return ImageFeature3D
   */
  def transform(feature: ImageFeature3D): ImageFeature3D = {
    try {
      if (!feature.isValid) return feature
      // change image to tensor
      val tensor = feature.asInstanceOf[ImageFeature3D][Tensor[Float]](ImageFeature.imageTensor)
      val out = transformTensor(tensor)
      feature.update(ImageFeature.imageTensor, out)
      feature.update(ImageFeature.size, out.size())
    } catch {
      case e: Exception =>
        feature.isValid = false
        if (ignoreImageException) {
          val path = if (feature.contains(ImageFeature.uri)) feature(ImageFeature.uri) else ""
          logger.warn(s"failed ${path} in transformer ${getClass}")
          e.printStackTrace()
        } else {
          throw e
        }

    }
    feature
  }


  override def apply(imageSet: ImageSet): ImageSet = {
    imageSet.transform(this)
  }

}

object ImageProcessing3D {
  val logger = LogManager.getLogger(getClass)
}
