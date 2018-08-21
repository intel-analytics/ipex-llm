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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.utils.T
import org.apache.log4j.Logger

import scala.collection.{Set, mutable}
import scala.reflect._

/**
 * Each ImageFeature keeps information about single image,
 * it can include various status of an image,
 * e.g. original bytes read from image file, an opencv mat,
 * pixels in float array, image label, meta data and so on.
 * it uses HashMap to store all these data,
 * the key is string that identify the corresponding value
 */
class ImageFeature extends Serializable {

  import ImageFeature.logger

  /**
   * Create ImageFeature from original image in byte array, label and uri
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

  /**
   * whether this image feature is valid
   */
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

  /**
   * whether this ImageFeature contains label
   * @return
   */
  def hasLabel(): Boolean = state.contains(ImageFeature.label)

  /**
   * image file in bytes
   */
  def bytes(): Array[Byte] = apply[Array[Byte]](ImageFeature.bytes)

  /**
   * get uri from ImageFeature
   * @return
   */
  def uri(): String = apply[String](ImageFeature.uri)

  /**
   * image pixels in float array
   *
   * @param key key that maps float array
   * @return float array
   */
  def floats(key: String = ImageFeature.floats): Array[Float] = {
    apply[Array[Float]](key)
  }

  /**
   * get prediction result from ImageFeature
   * @param key key that maps prediction result
   */
  def predict(key: String = ImageFeature.predict): Any = {
    apply(key)
  }

  /**
   * get current image size in (height, width, channel)
   *
   * @return (height, width, channel)
   */
  def getSize: (Int, Int, Int) = {
    val mat = opencvMat()
    if (mat != null && !mat.isReleased) {
      mat.shape()
    } else if (contains(ImageFeature.size)) {
      apply[(Int, Int, Int)](ImageFeature.size)
    } else {
      getOriginalSize
    }
  }

  /**
   * get current height
   */
  def getHeight(): Int = getSize._1

  /**
   * get current width
   */
  def getWidth(): Int = getSize._2

  /**
   * get current channel
   */
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

  /**
   * get original width
   */
  def getOriginalWidth: Int = getOriginalSize._2

  /**
   * get original height
   */
  def getOriginalHeight: Int = getOriginalSize._1

  /**
   * get original channel
   */
  def getOriginalChannel: Int = getOriginalSize._3

  /**
   * get label from ImageFeature
   */
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

  /**
   * clear ImageFeature
   */
  def clear(): Unit = {
    state.clear()
    isValid = true
  }

  override def clone(): ImageFeature = {
    val imageFeature = new ImageFeature()
    state.foreach(x => {
      imageFeature(x._1) = x._2
    })
    imageFeature.isValid = isValid
    imageFeature
  }


  /**
   * copy the float array to a storage
   * @param storage destination array
   * @param offset offset to copy
   * @param floatKey key that maps float array
   * @param toRGB BGR to RGB
   */
  def copyTo[T: ClassTag](storage: Array[T], offset: Int, floatKey: String = ImageFeature.floats,
    toRGB: Boolean = true, greyToRGB: Boolean = false)(implicit ev: TensorNumeric[T]): Unit = {
    val channel = getChannel()
    require(contains(floatKey), s"there should be ${floatKey} in ImageFeature")
    val data = floats(floatKey)
    require(data.length >= getWidth() * getHeight() * channel,
      s"float array length should be larger than $channel * ${getWidth()} * ${getHeight()}")
    val frameLength = getWidth() * getHeight()
    require(frameLength * channel + offset <= storage.length)
    if (channel == 3) {
      copyBGR(storage, offset, toRGB, data, frameLength)
    } else if (!greyToRGB) {
      copyChannels(storage, offset, channel, data, frameLength)
    } else {
      copyGreyToRGB(storage, offset, data, frameLength)
    }
  }

  private def copyBGR[T: ClassTag](storage: Array[T], offset: Int, toRGB: Boolean,
    data: Array[Float], frameLength: Int): Unit = {
    if (classTag[T] == classTag[Float]) {
      val storageFloat = storage.asInstanceOf[Array[Float]]
      var j = 0
      if (toRGB) {
        while (j < frameLength) {
          storageFloat(offset + j) = data(j * 3 + 2)
          storageFloat(offset + j + frameLength) = data(j * 3 + 1)
          storageFloat(offset + j + frameLength * 2) = data(j * 3)
          j += 1
        }
      } else {
        while (j < frameLength) {
          storageFloat(offset + j) = data(j * 3)
          storageFloat(offset + j + frameLength) = data(j * 3 + 1)
          storageFloat(offset + j + frameLength * 2) = data(j * 3 + 2)
          j += 1
        }
      }
    } else if (classTag[T] == classTag[Double]) {
      val storageDouble = storage.asInstanceOf[Array[Double]]
      var j = 0
      if (toRGB) {
        while (j < frameLength) {
          storageDouble(offset + j) = data(j * 3 + 2)
          storageDouble(offset + j + frameLength) = data(j * 3 + 1)
          storageDouble(offset + j + frameLength * 2) = data(j * 3)
          j += 1
        }
      } else {
        while (j < frameLength) {
          storageDouble(offset + j) = data(j * 3)
          storageDouble(offset + j + frameLength) = data(j * 3 + 1)
          storageDouble(offset + j + frameLength * 2) = data(j * 3 + 2)
          j += 1
        }
      }
    }
  }

  private def copyChannels[T: ClassTag](storage: Array[T], offset: Int, channel: Int,
    data: Array[Float], frameLength: Int): Unit = {
    if (classTag[T] == classTag[Float]) {
      val storageFloat = storage.asInstanceOf[Array[Float]]
      var j = 0
      while (j < frameLength) {
        var c = 0
        while (c < channel) {
          storageFloat(offset + j + frameLength * c) = data(j * channel + c)
          c += 1
        }
        j += 1
      }
    } else if (classTag[T] == classTag[Double]) {
      val storageDouble = storage.asInstanceOf[Array[Double]]
      var j = 0
      while (j < frameLength) {
        var c = 0
        while (c < channel) {
          storageDouble(offset + j + frameLength * c) = data(j * channel + c)
          c += 1
        }
        j += 1
      }
    }
  }

  private def copyGreyToRGB[T: ClassTag](storage: Array[T], offset: Int, data: Array[Float],
    frameLength: Int): Unit = {
    require(offset + frameLength * 3 <= storage.length,
      s"tensor storage cannot hold the whole image data, offset $offset " +
        s"data length ${data.length} storage lenght ${storage.length}")
    if (classTag[T] == classTag[Float]) {
      val storageFloat = storage.asInstanceOf[Array[Float]]
      var c = 0
      while(c < 3) {
        var i = 0
        while(i < frameLength) {
          storageFloat(i + c * frameLength + offset) = data(i)
          i += 1
        }
        c += 1
      }
    } else {
      throw new IllegalArgumentException("Not support type")
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
      OpenCVMat.toFloatPixels(mat, floats)
      (floats, Array(mat.height(), mat.width(), 3))
    }
    var image = Tensor(Storage(data)).resize(size)
    if (toChw) {
      // transpose the shape of image from (h, w, c) to (c, h, w)
      image = image.transpose(1, 3).transpose(2, 3).contiguous()
    }
    image
  }

  /**
   * set label for imagefeature from label map
   */

  def setLabel(labelMap: mutable.Map[String, Float]): Unit = {
    val uri = this.uri
    if (labelMap.contains(uri)) {
      this(ImageFeature.label) = Tensor[Float](T(labelMap(uri)))
    }
  }
}

object ImageFeature {
  /**
   * key: uri that identifies image
   */
  val uri = "uri"
  /**
   * key: image in OpenCVMat
   */
  val mat = "mat"
  /**
   * key: image file in bytes
   */
  val bytes = "bytes"
  /**
   * key: image pixels in float array
   */
  val floats = "floats"
  /**
   * key: current image size
   */
  val size = "size"
  /**
   * key: original image size
   */
  val originalSize = "originalSize"
  /**
   * key: label
   */
  val label = "label"
  /**
   * key: image prediction result
   */
  val predict = "predict"
  /**
   * key: store boundingBox of current image
   * it may be used in crop/expand that may change the size of image
   */
  val boundingBox = "boundingBox"

  /**
   * key: image (and label if available) stored as Sample
   */
  val sample = "sample"

  /**
   * key: image pixels in Tensor
   */
  val imageTensor = "imageTensor"

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
