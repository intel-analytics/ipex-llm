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

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.utils.T
import org.apache.log4j.Logger

import scala.collection.{Set, mutable}
import scala.reflect.ClassTag

class ImageFeature extends Serializable {

  import ImageFeature.logger

  /**
   * Create ImageFeature
   *
   * @param bytes image file in bytes
   * @param label label
   * @param uri image uri
   */
  def this(bytes: Array[Byte], label: Any = null, uri: String = null) {
    this
    state(ImageFeature.bytes) = bytes
    if (null != uri) {
      state(ImageFeature.uri) = uri
    }
    if (null != label) {
      state(ImageFeature.label) = label
    }
  }

  private val state = new mutable.HashMap[String, Any]()

  var isValid = true

  def apply[T](key: String): T = {
    if (contains(key)) state(key).asInstanceOf[T] else null.asInstanceOf[T]
  }

  def update(key: String, value: Any): Unit = state(key) = value

  def contains(key: String): Boolean = state.contains(key)

  /**
   * get opencv mat from ImageFeature, note that it may be empty if it is released
   */
  def opencvMat(): OpenCVMat = apply[OpenCVMat](ImageFeature.mat)

  def keys(): Set[String] = state.keySet

  def hasLabel(): Boolean = state.contains(ImageFeature.label)

  /**
   * image file in bytes
   */
  def bytes(): Array[Byte] = apply[Array[Byte]](ImageFeature.bytes)

  def uri(): String = apply[String](ImageFeature.uri)

  /**
   * image pixels in float array
   *
   * @param key key that map float array
   * @return float array
   */
  def floats(key: String = ImageFeature.floats): Array[Float] = {
    apply[Array[Float]](key)
  }

  def feature(key: String = ImageFeature.predict): Any = {
    apply(key)
  }

  def tensorFeature(key: String = ImageFeature.predict): Tensor[Float] = {
    feature(key).asInstanceOf[Tensor[Float]]
  }

  /**
   * get current image size in (height, width, channel)
   *
   * @return (height, width, channel)
   */
  def getSize: (Int, Int, Int) = {
    val mat = opencvMat()
    if (!mat.isReleased) {
      mat.shape()
    } else if (contains(ImageFeature.size)) {
      apply[(Int, Int, Int)](ImageFeature.size)
    } else {
      getOriginalSize
    }
  }

  def getHeight(): Int = getSize._1

  def getWidth(): Int = getSize._2

  def getChannel(): Int = getSize._3

  /**
   * get original image size in (height, width, channel)
   *
   * @return (height, width, channel)
   */
  def getOriginalSize: (Int, Int, Int) = {
    if (contains(ImageFeature.originalSize)) {
      apply[(Int, Int, Int)](ImageFeature.originalSize)
    } else {
      logger.warn("there is no original size stored")
      (-1, -1, -1)
    }
  }

  def getOriginalWidth: Int = getOriginalSize._2

  def getOriginalHeight: Int = getOriginalSize._1

  def getLabel[T: ClassTag]: T = apply[T](ImageFeature.label)

  /**
   * imInfo is a tensor that contains height, width, scaleInHeight, scaleInWidth
   * e.g. it is used in SSD and Faster-RCNN to post process the roi detection
   */
  def getImInfo(): Tensor[Float] = {
    val (height, width, _) = getSize
    val (oh, ow, _) = getOriginalSize
    Tensor[Float](T(height, width, height.toFloat / oh, width.toFloat / ow))
  }

  def clear(): Unit = {
    state.clear()
    isValid = true
  }


  def copyTo(storage: Array[Float], offset: Int, floatKey: String = ImageFeature.floats,
             toRGB: Boolean = true): Unit = {
    require(contains(floatKey), s"there should be ${floatKey} in ImageFeature")
    val data = floats(floatKey)
    require(data.length >= getWidth() * getHeight() * 3,
      s"float array length should be larger than 3 * ${getWidth()} * ${getHeight()}")
    val frameLength = getWidth() * getHeight()
    require(frameLength * 3 + offset <= storage.length)
    var j = 0
    if (toRGB) {
      while (j < frameLength) {
        storage(offset + j) = data(j * 3 + 2)
        storage(offset + j + frameLength) = data(j * 3 + 1)
        storage(offset + j + frameLength * 2) = data(j * 3)
        j += 1
      }
    } else {
      while (j < frameLength) {
        storage(offset + j) = data(j * 3)
        storage(offset + j + frameLength) = data(j * 3 + 1)
        storage(offset + j + frameLength * 2) = data(j * 3 + 2)
        j += 1
      }
    }
  }

  /**
   * Convert ImageFeature to image tensor
   * @param floatKey key that maps the float array
   * @param toChw transpose the image from hwc to chw
   * @return tensor that represents an image
   */
  def toTensor(floatKey: String, toChw: Boolean = true): Tensor[Float] = {
    val (data, size) = if (contains(floatKey)) {
      (floats(floatKey),
        Array(getHeight(), getWidth(), 3))
    } else {
      logger.warn(s"please add MatToFloats(out_key = $floatKey) in the end of pipeline if you" +
        s" are transforming an rdd")
      val mat = opencvMat()
      val floats = new Array[Float](mat.height() * mat.width() * 3)
      OpenCVMat.toFloatBuf(mat, floats)
      (floats, Array(mat.height(), mat.width(), 3))
    }
    var image = Tensor(Storage(data)).resize(size)
    if (toChw) {
      // transpose the shape of image from (h, w, c) to (c, h, w)
      image = image.transpose(1, 3).transpose(2, 3).contiguous()
    }
    image
  }
}

object ImageFeature {
  val label = "label"
  val uri = "uri"
  // image in OpenCVMat
  val mat = "mat"
  // image file in bytes
  val bytes = "bytes"
  // image pixels in float array
  val floats = "floats"
  // current image size
  val size = "size"
  // original image size
  val originalSize = "originalSize"
  // image prediction result
  val predict = "predict"
  val cropBbox = "cropBbox"
  val expandBbox = "expandBbox"

  /**
   * Create ImageFeature
   *
   * @param bytes image file in bytes
   * @param label label
   * @param uri image uri
   */
  def apply(bytes: Array[Byte], label: Any = null, uri: String = null)
  : ImageFeature = new ImageFeature(bytes, label, uri)

  def apply(): ImageFeature = new ImageFeature()

  val logger = Logger.getLogger(getClass)
}