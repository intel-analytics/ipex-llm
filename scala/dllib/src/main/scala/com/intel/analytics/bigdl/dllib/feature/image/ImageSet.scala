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

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.transform.vision.image.{DistributedImageFrame, ImageFeature, ImageFrame, LocalImageFrame}
import com.intel.analytics.zoo.common.Utils
import com.intel.analytics.zoo.feature.common.Preprocessing
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.opencv.imgcodecs.Imgcodecs

/**
 * ImageSet wraps a set of ImageFeature
 */
abstract class ImageSet {
  /**
   * transform ImageSet
   * @param transformer FeatureTransformer
   * @return transformed ImageSet
   */
  def transform(transformer: Preprocessing[ImageFeature, ImageFeature]): ImageSet

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> (transformer: Preprocessing[ImageFeature, ImageFeature]): ImageSet = {
    this.transform(transformer)
  }

  /**
   * whether this is a LocalImageSet
   */
  def isLocal(): Boolean

  /**
   * whether this is a DistributedImageSet
   */
  def isDistributed(): Boolean

  /**
   * return LocalImageSet
   */
  def toLocal(): LocalImageSet = this.asInstanceOf[LocalImageSet]

  /**
   * return DistributedImageSet
   */
  def toDistributed(): DistributedImageSet = this.asInstanceOf[DistributedImageSet]


  /**
   * Convert ImageFrame to ImageSet
   *
   * @return ImageSet
   */
  def toImageFrame(): ImageFrame

  /**
   * Convert ImageSet to DataSet of ImageFeature.
   */
  def toDataSet(): DataSet[ImageFeature]
}

class LocalImageSet(var array: Array[ImageFeature]) extends ImageSet {
  override def transform(transformer: Preprocessing[ImageFeature, ImageFeature]): ImageSet = {
    array = transformer.apply(array.toIterator).toArray
    this
  }

  override def isLocal(): Boolean = true

  override def isDistributed(): Boolean = false

  override def toImageFrame(): ImageFrame = {
    ImageFrame.array(array)
  }

  override def toDataSet(): DataSet[ImageFeature] = {
    DataSet.array(array)
  }
}

class DistributedImageSet(var rdd: RDD[ImageFeature]) extends ImageSet {
  override def transform(transformer: Preprocessing[ImageFeature, ImageFeature]): ImageSet = {
    rdd = transformer(rdd)
    this
  }

  override def isLocal(): Boolean = false

  override def isDistributed(): Boolean = true

  override def toImageFrame(): ImageFrame = {
    ImageFrame.rdd(rdd)
  }

  override def toDataSet(): DataSet[ImageFeature] = {
    DataSet.rdd[ImageFeature](rdd)
  }
}

object ImageSet {

  /**
   * create LocalImageSet
   * @param data array of ImageFeature
   */
  def array(data: Array[ImageFeature]): LocalImageSet = {
    new LocalImageSet(data)
  }

  /**
   * create LocalImageSet from array of bytes
   * @param data nested array of bytes, expect inner array is a image
   */
  def array(data: Array[Array[Byte]]): LocalImageSet = {
    val images = data.map(ImageFeature(_))
    ImageSet.array(images)
  }

  /**
   * create DistributedImageSet
   * @param data rdd of ImageFeature
   */
  def rdd(data: RDD[ImageFeature]): DistributedImageSet = {
    new DistributedImageSet(data)
  }

  /**
   * create DistributedImageSet for a RDD of array bytes
   * @param data rdd of array of bytes
   */
  def rddBytes(data: RDD[Array[Byte]]): DistributedImageSet = {
    val images = data.map(ImageFeature(_))
    ImageSet.rdd(images)
  }

  /**
   * Read images as Image Set
   * if sc is defined, Read image as DistributedImageSet from local file system or HDFS
   * if sc is null, Read image as LocalImageSet from local file system
   *
   * @param path path to read images
   * if sc is defined, path can be local or HDFS. Wildcard character are supported.
   * if sc is null, path is local directory/image file/image file with wildcard character
   * @param sc SparkContext
   * @param minPartitions A suggestion value of the minimal splitting number for input data.
   * @param resizeH height after resize, by default is -1 which will not resize the image
   * @param resizeW width after resize, by default is -1 which will not resize the image
   * @param imageCodec specifying the color type of a loaded image, same as in OpenCV.imread.
   *              By default is Imgcodecs.CV_LOAD_IMAGE_UNCHANGED
   * @return ImageSet
   */
  def read(path: String, sc: SparkContext = null, minPartitions: Int = 1,
           resizeH: Int = -1, resizeW: Int = -1,
           imageCodec: Int = Imgcodecs.CV_LOAD_IMAGE_UNCHANGED): ImageSet = {
    val imageSet = if (null != sc) {
      val images = sc.binaryFiles(path, minPartitions).map { case (p, stream) =>
          ImageFeature(stream.toArray(), uri = p)
      }
      ImageSet.rdd(images)
    } else {
      val files = Utils.listLocalFiles(path)
      val images = files.map { p =>
        ImageFeature(FileUtils.readFileToByteArray(p), uri = p.getAbsolutePath)
      }
      ImageSet.array(images)
    }
    if (resizeW == -1 || resizeH == -1) {
      imageSet -> ImageBytesToMat(imageCodec = imageCodec)
    } else {
      imageSet -> BufferedImageResize(resizeH, resizeW) -> ImageBytesToMat(imageCodec = imageCodec)
    }
  }

  /**
   * Convert ImageFrame to ImageSet
   *
   * @param imageFrame imageFrame which needs to covert to Imageset
   * @return ImageSet
   */
  private[zoo] def fromImageFrame(imageFrame: ImageFrame): ImageSet = {
    val imageset = imageFrame match {
      case distributedImageFrame: DistributedImageFrame =>
        ImageSet.rdd(imageFrame.toDistributed().rdd)
      case localImageFrame: LocalImageFrame =>
        ImageSet.array(imageFrame.toLocal().array)
    }
    imageset
  }
}
