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
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature.getClass
import org.apache.log4j.Logger


/**
 * ImageFeature3D keeps information about single 3D image,
 * it can include various status of an image,
 * e.g. multi-dimension array of float, image label, meta data and so on.
 * it uses HashMap to store all these data,
 * the key is string that identify the corresponding value
 */
class ImageFeature3D extends ImageFeature {

  import ImageFeature3D.logger

  def this(tensor: Tensor[Float], label: Any = null, uri: String = null) {
    this
    update(ImageFeature.imageTensor, tensor)
    update(ImageFeature.size, tensor.size)
    if (null != uri) {
      update(ImageFeature.uri, uri)
    }
    if (null != label) {
      update(ImageFeature.label, label)
    }
  }

  /**
   * get current image size in [depth, height, width, channel]
   *
   * @return size array: [depth, height, width, channel]
   */
  def getImageSize(): Array[Int] = {
    apply[Array[Int]](ImageFeature.size)
  }

  override def getSize: (Int, Int, Int) = {
    logger.warn("this function is deprecated in ImageFeature3D")
    null
  }

  /**
   * get current height
   */
  def getDepth(): Int = getImageSize()(0)

  /**
   * get current height
   */
  override def getHeight(): Int = getImageSize()(1)

  /**
   * get current width
   */
  override def getWidth(): Int = getImageSize()(2)

  /**
   * get current channel
   */
  override def getChannel(): Int = {
    logger.warn("Currrently 3D image only suport 1 channel")
    1
  }


  override def clone(): ImageFeature3D = {
    val imageFeature = new ImageFeature3D()
    keys().foreach(key => {
      imageFeature.update(key, this.apply(key))
    })
    imageFeature.isValid = isValid
    imageFeature
  }
}


object ImageFeature3D {

  def apply(tensor: Tensor[Float], uri: String = null, label: Any = null)
  : ImageFeature3D = new ImageFeature3D(tensor, label, uri)

  def apply(): ImageFeature3D = new ImageFeature3D()

  val logger = Logger.getLogger(getClass)
}

