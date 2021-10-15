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

import com.intel.analytics.bigdl.dllib.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.{ImageFeature, MatToFloats, MatToTensor}
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image

import scala.reflect.ClassTag

/**
 * transform opencv mat to tensor
 * @param toRGB BGR to RGB (default is BGR)
 * @param tensorKey key to store transformed tensor
 * @param format DataFormat.NCHW or DataFormat.NHWC
 */
class ImageMatToTensor[T: ClassTag](
    toRGB: Boolean = false,
    tensorKey: String = ImageFeature.imageTensor,
    shareBuffer: Boolean = true,
    format: DataFormat = DataFormat.NCHW)(implicit ev: TensorNumeric[T])
  extends ImageProcessing {
  private val imageTensor: Tensor[T] = Tensor[T]()
  private val matToFloats = MatToFloats()

  private val internalMatToTensor = new image.MatToTensor[T](toRGB, tensorKey, shareBuffer)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    prev.map(transform(_))
  }

  override def transform(feature: ImageFeature): ImageFeature = {
    format match {
      case DataFormat.NHWC => transformWithNHWC(feature)
      case DataFormat.NCHW => internalMatToTensor.transform(feature)
      case other => throw new IllegalArgumentException(s"Unsupported format:" +
        s" $format. Only NCHW and NHWC are supported.")
    }
  }

  def transformWithNHWC(feature: ImageFeature): ImageFeature = {
    if (!feature.isValid) return feature
    try {
      val (height, width, channel) = feature.getSize
      matToFloats.transform(feature)
      if (channel == 1) {
        imageTensor.resize(height, width)
      } else {
        imageTensor.resize(height, width, channel)
      }
      System.arraycopy(feature.floats(), 0, imageTensor.storage().array(),
        0, imageTensor.nElement())
      if (!shareBuffer) {
        feature(tensorKey) = imageTensor.clone()
      } else {
        feature(tensorKey) = imageTensor
      }
    } catch {
      case e: Exception =>
        val uri = feature.uri()
        MatToTensor.logger.warn(s"float to tensor fail for ${uri}")
        e.printStackTrace()
        feature.isValid = false
    }
    feature
  }
}

object ImageMatToTensor {
  def apply[T: ClassTag](
      toRGB: Boolean = false,
      tensorKey: String = ImageFeature.imageTensor,
      shareBuffer: Boolean = true,
      format: DataFormat = DataFormat.NCHW
  )(implicit ev: TensorNumeric[T]): ImageMatToTensor[T] =
    new ImageMatToTensor[T](toRGB, tensorKey, shareBuffer, format)
}
