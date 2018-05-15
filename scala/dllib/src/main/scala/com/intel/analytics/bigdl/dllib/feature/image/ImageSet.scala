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

import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.zoo.common.Utils

import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
 * ImageSet wraps a set of ImageFeature
 */
abstract class ImageSet {
  /**
   * transform ImageSet
   * @param transformer FeatureTransformer
   * @return transformed ImageSet
   */
  def transform(transformer: FeatureTransformer): ImageSet

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> (transformer: FeatureTransformer): ImageSet = {
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
}

/**
 * Local ImageSet, keeps an array of ImageFeature
 * @param array array of ImageFeature
 */
class LocalImageSet(var array: Array[ImageFeature]) extends ImageSet {
  override def transform(transformer: FeatureTransformer): ImageSet = {
    array = array.map(transformer.transform)
    this
  }

  override def isLocal(): Boolean = true

  override def isDistributed(): Boolean = false

  override def toImageFrame(): ImageFrame = {
    ImageFrame.array(array)
  }
}

/**
 * Distributerd ImageSet, it keeps an rdd of ImageFeature
 * @param rdd rdd of ImageFeature
 */
class DistributedImageSet(var rdd: RDD[ImageFeature]) extends ImageSet {
  override def transform(transformer: FeatureTransformer): ImageSet = {
    rdd = transformer(rdd)
    this
  }

  override def isLocal(): Boolean = false

  override def isDistributed(): Boolean = true

  override def toImageFrame(): ImageFrame = {
    ImageFrame.rdd(rdd)
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
   * create DistributedImageSet
   * @param data rdd of ImageFeature
   */
  def rdd(data: RDD[ImageFeature]): DistributedImageSet = {
    new DistributedImageSet(data)
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
   * @return ImageSet
   */
  def read(path: String, sc: SparkContext = null, minPartitions: Int = 1): ImageSet = {
    if (null != sc) {
      val images = sc.binaryFiles(path, minPartitions).map { case (p, stream) =>
        ImageFeature(stream.toArray(), uri = p)
      }
      ImageSet.rdd(images) -> BytesToMat()
    } else {
      val files = Utils.listLocalFiles(path)
      val images = files.map { p =>
        ImageFeature(FileUtils.readFileToByteArray(p), uri = p.getAbsolutePath)
      }
      ImageSet.array(images) -> BytesToMat()
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
