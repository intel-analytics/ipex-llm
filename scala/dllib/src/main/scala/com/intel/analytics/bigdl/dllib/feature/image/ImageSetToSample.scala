/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.feature.image

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrameToSample}

import scala.reflect.ClassTag

/**
 * Transforms tensors that map inputKeys and targetKeys to sample
 * @param inputKeys keys that maps inputs (each input should be a tensor)
 * @param targetKeys keys that maps targets (each target should be a tensor)
 * @param sampleKey key to store sample
 */
class ImageSetToSample[T: ClassTag](inputKeys: Array[String] = Array(ImageFeature.imageTensor),
                       targetKeys: Array[String] = null,
                       sampleKey: String = ImageFeature.sample)(implicit ev: TensorNumeric[T])
  extends ImageProcessing {
  private val internalCrop = ImageFrameToSample[T](inputKeys, targetKeys, sampleKey)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalCrop.apply(prev)
  }
}

object ImageSetToSample {
  def apply[T: ClassTag](inputKeys: Array[String] = Array(ImageFeature.imageTensor),
            targetKeys: Array[String] = null,
            sampleKey: String = ImageFeature.sample)
            (implicit ev: TensorNumeric[T]): ImageSetToSample[T] =
    new ImageSetToSample(inputKeys, targetKeys, sampleKey)
}
