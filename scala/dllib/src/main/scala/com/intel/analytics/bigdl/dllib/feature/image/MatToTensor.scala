package com.intel.analytics.zoo.feature.image

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature}
import com.intel.analytics.bigdl.transform.vision.image
import com.intel.analytics.zoo.feature.common.Preprocessing

import scala.reflect.ClassTag

class MatToTensor[T: ClassTag](
    toRGB: Boolean = false,
    tensorKey: String = ImageFeature.imageTensor,
    shareBuffer: Boolean = true)(implicit ev: TensorNumeric[T])
  extends Preprocessing[ImageFeature, ImageFeature] {
  
  private val internalResize = new image.MatToTensor[T](toRGB, tensorKey, shareBuffer)
  def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalResize.apply(prev)
  }
}

object MatToTensor {

  def apply[T: ClassTag](
      toRGB: Boolean = false,
      tensorKey: String = ImageFeature.imageTensor,
      shareBuffer: Boolean = true
  )(implicit ev: TensorNumeric[T]): MatToTensor[T] = new MatToTensor[T](toRGB, tensorKey, shareBuffer)
}
