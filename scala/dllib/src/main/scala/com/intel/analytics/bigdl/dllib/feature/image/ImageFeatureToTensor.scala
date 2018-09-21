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

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature._
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.feature.common.Preprocessing

import scala.reflect.ClassTag

/**
 * a Preprocessing that convert ImageFeature to a Tensor.
 */
class ImageFeatureToTensor[T: ClassTag](toChw: Boolean = true)(implicit ev: TensorNumeric[T])
  extends Preprocessing[ImageFeature, Tensor[T]] {

  override def apply(prev: Iterator[ImageFeature]): Iterator[Tensor[T]] = {
    prev.map { imf =>
      if (imf.contains(ImageFeature.imageTensor)) {
        val tensor = imf(ImageFeature.imageTensor).asInstanceOf[Tensor[T]]
        val img = if (tensor.dim() !=2 && (toChw && tensor.size(1) > tensor.size(3)
          || !toChw && tensor.size(1) < tensor.size(3))) {
          tensor.transpose(1, 3).transpose(1, 2).contiguous()
        } else tensor
        img
      } else {
        toTensor(imf, toChw).asInstanceOf[Tensor[T]]
      }
    }
  }

  def toTensor(imf: ImageFeature, toChw: Boolean = true): Tensor[Float] = {
    val (data, size) = if (imf.contains(ImageFeature.floats)) {
      (imf.floats(),
        Array(imf.getHeight(), imf.getWidth(), imf.getChannel()))
    } else {
      logger.warn(s"please add MatToFloats(out_key =" +
        s"imf(ImageFeature.imageTensor).asInstanceOf[Tensor[T]])" +
        s"in the end of pipeline if you are transforming an rdd")
      val mat = imf.opencvMat()
      val floats = new Array[Float](mat.height() * mat.width() * imf.getChannel())
      OpenCVMat.toFloatPixels(mat, floats)
      (floats, Array(mat.height(), mat.width(), imf.getChannel()))
    }
    var image = Tensor(Storage(data)).resize(size)
    if (toChw) {
      // transpose the shape of image from (h, w, c) to (c, h, w)
      image = image.transpose(1, 3).transpose(2, 3).contiguous()
    }
    image
  }
}

object ImageFeatureToTensor {
  def apply[T: ClassTag](toChw: Boolean = true)
    (implicit ev: TensorNumeric[T]): ImageFeatureToTensor[T] =
    new ImageFeatureToTensor[T](toChw)
}

