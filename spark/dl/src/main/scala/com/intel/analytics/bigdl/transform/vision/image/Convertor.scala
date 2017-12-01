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

package com.intel.analytics.bigdl.transform.vision.image

import com.intel.analytics.bigdl.dataset.ArraySample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import org.apache.log4j.Logger

import scala.reflect._

/**
 * Transform byte array(original image file in byte) to OpenCVMat
 */
class BytesToMat()
  extends FeatureTransformer {

  override def transform(feature: ImageFeature): ImageFeature = {
    BytesToMat.transform(feature)
  }
}

object BytesToMat {
  val logger = Logger.getLogger(getClass)
  def apply(): BytesToMat = new BytesToMat()

  def transform(feature: ImageFeature): ImageFeature = {
    if (!feature.isValid) return feature
    val bytes = feature[Array[Byte]](ImageFeature.bytes)
    var mat: OpenCVMat = null
    try {
      require(null != bytes && bytes.length > 0, "image file bytes should not be empty")
      mat = OpenCVMat.fromImageBytes(bytes)
      feature(ImageFeature.mat) = mat
      feature(ImageFeature.originalSize) = mat.shape()
    } catch {
      case e: Exception =>
        val uri = feature.uri()
        logger.warn(s"convert byte to mat fail for $uri")
        feature(ImageFeature.originalSize) = (-1, -1, -1)
        feature.isValid = false
    }
    feature
  }
}


/**
 * Transform OpenCVMat to float array, note that in this transformer, the mat is released
 * @param validHeight valid height in case the mat is invalid
 * @param validWidth valid width in case the mat is invalid
 * @param validChannels valid channel in case the mat is invalid
 * @param meanRGB meansRGB to subtract, it can be replaced by ChannelNormalize
 * @param outKey key to store float array
 */
class MatToFloats(validHeight: Int, validWidth: Int, validChannels: Int,
  meanRGB: Option[(Float, Float, Float)] = None, outKey: String = ImageFeature.floats)
  extends FeatureTransformer {
  @transient private var data: Array[Float] = _

  private def normalize(img: Array[Float],
    meanR: Float, meanG: Float, meanB: Float): Array[Float] = {
    val content = img
    require(content.length % 3 == 0)
    var i = 0
    while (i < content.length) {
      content(i + 2) = content(i + 2) - meanR
      content(i + 1) = content(i + 1) - meanG
      content(i + 0) = content(i + 0) - meanB
      i += 3
    }
    img
  }

  override def transform(feature: ImageFeature): ImageFeature = {
    var input: OpenCVMat = null
    val (height, width, channel) = if (feature.isValid) {
      input = feature.opencvMat()
      (input.height(), input.width(), input.channels())
    } else {
      (validHeight, validWidth, validChannels)
    }
    if (null == data || data.length < height * width * channel) {
      data = new Array[Float](height * width * channel)
    }
    if (feature.isValid) {
      try {
        OpenCVMat.toFloatPixels(input, data)
        if (meanRGB.isDefined) {
          normalize(data, meanRGB.get._1, meanRGB.get._2, meanRGB.get._3)
        }
      } finally {
        if (null != input) input.release()
      }
    }
    feature(outKey) = data
    feature(ImageFeature.size) = (height, width, channel)
    feature
  }
}

object MatToFloats {
  val logger = Logger.getLogger(getClass)

  def apply(validHeight: Int = 300, validWidth: Int = 300, validChannels: Int = 3,
    meanRGB: Option[(Float, Float, Float)] = None,
    outKey: String = ImageFeature.floats): MatToFloats =
    new MatToFloats(validHeight, validWidth, validChannels, meanRGB, outKey)
}

class FloatsToTensor[T: ClassTag](toRGB: Boolean = false,
  floatKey: String = ImageFeature.floats,
  tensorKey: String = ImageFeature.imageTensor)(implicit ev: TensorNumeric[T])
  extends FeatureTransformer {
  private val imageTensor = Tensor[T]()

  override def transform(feature: ImageFeature): ImageFeature = {
    if (!feature.isValid) return feature
    try {
      imageTensor.resize(3, feature.getHeight(), feature.getWidth())
      feature.copyTo[T](imageTensor.storage().array(), 0, floatKey, toRGB)
      feature(tensorKey) = imageTensor
    } catch {
      case e: Exception =>
        val uri = feature.uri()
        FloatsToTensor.logger.warn(s"float to tensor fail for ${uri}")
        feature.isValid = false
    }
    feature
  }
}

object FloatsToTensor {
  val logger = Logger.getLogger(getClass)

  def apply[T: ClassTag](toRGB: Boolean = false, floatKey: String = ImageFeature.floats,
    tensorKey: String = ImageFeature.imageTensor)(implicit ev: TensorNumeric[T])
  : FloatsToTensor[T] = new FloatsToTensor[T](toRGB, floatKey, tensorKey)
}

class ImageFrameToSample[T: ClassTag](inputKeys: Array[String] = Array(ImageFeature.imageTensor),
  targetKeys: Array[String] = null,
  sampleKey: String = ImageFeature.sample)
  (implicit ev: TensorNumeric[T]) extends FeatureTransformer {


  override def transform(feature: ImageFeature): ImageFeature = {
    if (!feature.isValid) return feature
    try {
      val inputs = inputKeys.map(key => {
        val input = feature(key)
        require(input.isInstanceOf[Tensor[T]], s"the input $key should be tensor")
        input.asInstanceOf[Tensor[T]]
      })
      val sample = if (targetKeys == null) {
        ArraySample[T](inputs)
      } else {
        val targets = targetKeys.map(key => {
          val target = feature(key)
          require(target.isInstanceOf[Tensor[T]], s"the target $key should be tensor")
          target.asInstanceOf[Tensor[T]]
        })
        ArraySample[T](inputs, targets)
      }
      feature(sampleKey) = sample
    } catch {
      case e: Exception =>
        val uri = feature.uri()
        ImageFrameToSample.logger.warn(s"convert byte to mat fail for $uri")
        feature(ImageFeature.originalSize) = (-1, -1, -1)
        feature.isValid = false
    }
    feature
  }
}

object ImageFrameToSample {
  val logger = Logger.getLogger(getClass)

  def apply[T: ClassTag](inputKeys: Array[String] = Array(ImageFeature.imageTensor),
    targetKeys: Array[String] = null,
    sampleKey: String = ImageFeature.sample)(implicit ev: TensorNumeric[T])
  : ImageFrameToSample[T] = new ImageFrameToSample[T](inputKeys, targetKeys, sampleKey)
}
