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

import java.io.{File, FilenameFilter}

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.commons.io.FileUtils
import org.apache.commons.io.filefilter.WildcardFileFilter
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * ImageFrame wraps a set of ImageFeature
 */
trait ImageFrame extends Serializable {

  /**
   * transform ImageFrame
   * @param transformer FeatureTransformer
   * @return transformed ImageFrame
   */
  def transform(transformer: FeatureTransformer): ImageFrame

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> (transformer: FeatureTransformer): ImageFrame = {
    this.transform(transformer)
  }

  /**
   * whether this is a LocalImageFrame
   */
  def isLocal(): Boolean

  /**
   * whether this is a DistributedImageFrame
   */
  def isDistributed(): Boolean

  /**
   * return LocalImageFrame
   */
  def toLocal(): LocalImageFrame = this.asInstanceOf[LocalImageFrame]

  /**
   * return DistributedImageFrame
   */
  def toDistributed(): DistributedImageFrame = this.asInstanceOf[DistributedImageFrame]

  /**
   * set label for ImageFrame
   * @param labelMap label map : uri to label mapping
   */

  def setLabel(labelMap: mutable.Map[String, Float]): Unit

}

object ImageFrame {
  val logger = Logger.getLogger(getClass)

  /**
   * create LocalImageFrame
   * @param data array of ImageFeature
   */
  def array(data: Array[ImageFeature]): LocalImageFrame = {
    new LocalImageFrame(data)
  }

  /**
   * create DistributedImageFrame
   * @param data rdd of ImageFeature
   */
  def rdd(data: RDD[ImageFeature]): DistributedImageFrame = {
    new DistributedImageFrame(data)
  }

  /**
   * Read images as Image Frame
   * if sc is defined, Read image as DistributedImageFrame from local file system or HDFS
   * if sc is null, Read image as LocalImageFrame from local file system
   *
   * @param path path to read images
   * if sc is defined, path can be local or HDFS. Wildcard character are supported.
   * if sc is null, path is local directory/image file/image file with wildcard character
   * @param sc SparkContext
   * @param minPartitions A suggestion value of the minimal splitting number for input data.
   * @return ImageFrame
   */
  def read(path: String, sc: SparkContext = null, minPartitions: Int = 1): ImageFrame = {
    if (null != sc) {
      val images = sc.binaryFiles(path, minPartitions).map { case (p, stream) =>
        ImageFeature(stream.toArray(), uri = p)
      }
      ImageFrame.rdd(images) -> BytesToMat()
    } else {
      val files = listLocalFiles(path)
      val images = files.map { p =>
        ImageFeature(FileUtils.readFileToByteArray(p), uri = p.getAbsolutePath)
      }
      ImageFrame.array(images) -> BytesToMat()
    }
  }

  private def listLocalFiles(path: String): Array[File] = {
    val files = new ArrayBuffer[File]()
    listFiles(path, files)
    files.toArray
  }

  private def listFiles(path: String, files: ArrayBuffer[File]): Unit = {
    val file = new File(path)
    if (file.isDirectory) {
      file.listFiles().foreach(x => listFiles(x.getAbsolutePath, files))
    } else if (file.isFile) {
      files.append(file)
    } else {
      val filter = new WildcardFileFilter(file.getName)
      file.getParentFile.listFiles(new FilenameFilter {
        override def accept(dir: File, name: String): Boolean = filter.accept(dir, name)
      }).foreach(x => listFiles(x.getAbsolutePath, files))
    }
  }


  /**
   * Read parquet file as DistributedImageFrame
   *
   * @param path Parquet file path
   * @return DistributedImageFrame
   */
  def readParquet(path: String, sqlContext: SQLContext): DistributedImageFrame = {
    val df = sqlContext.read.parquet(path)
    val images = df.rdd.map(row => {
      val uri = row.getAs[String](ImageFeature.uri)
      val image = row.getAs[Array[Byte]](ImageFeature.bytes)
      ImageFeature(image, uri = uri)
    })
    (ImageFrame.rdd(images) -> BytesToMat()).toDistributed()
  }

  /**
   * Write images as parquet file
   *
   * @param path path to read images. Local or HDFS. Wildcard character are supported.
   * @param output Parquet file path
   * @param partitionNum partition number
   */
  def writeParquet(path: String, output: String, sqlContext: SQLContext,
    partitionNum: Int = 1): Unit = {
    import sqlContext.implicits._
    val df = sqlContext.sparkContext.binaryFiles(path, partitionNum)
      .map { case (p, stream) =>
        (p, stream.toArray())
      }.toDF(ImageFeature.uri, ImageFeature.bytes)
    df.write.parquet(output)
  }
}

/**
 * Local ImageFrame, keeps an array of ImageFeature
 * @param array array of ImageFeature
 */
class LocalImageFrame(var array: Array[ImageFeature]) extends ImageFrame {

  def toDistributed(sc: SparkContext): DistributedImageFrame = {
    new DistributedImageFrame(sc.parallelize(array))
  }

  override def transform(transformer: FeatureTransformer): ImageFrame = {
    array = array.map(transformer.transform)
    this
  }

  override def isLocal(): Boolean = true

  override def isDistributed(): Boolean = false

  override def setLabel(labelMap: mutable.Map[String, Float]): Unit = {
    array = array.map(imageFeature => {
      imageFeature.setLabel(labelMap)
      imageFeature
    })
  }
}

/**
 * Distributerd ImageFrame, it keeps an rdd of ImageFeature
 * @param rdd rdd of ImageFeature
 */
class DistributedImageFrame(var rdd: RDD[ImageFeature]) extends ImageFrame {

  override def transform(transformer: FeatureTransformer): ImageFrame = {
    rdd = transformer(rdd)
    this
  }

  override def isLocal(): Boolean = false

  override def isDistributed(): Boolean = true

  override def setLabel(labelMap: mutable.Map[String, Float]): Unit = {
    rdd = rdd.map(imageFeature => {
      imageFeature.setLabel(labelMap)
      imageFeature
    })
  }

  def randomSplit(weights: Array[Double]): Array[ImageFrame] = {
    val splitRDD = rdd.randomSplit(weights)
    splitRDD.map(new DistributedImageFrame(_))
  }
}
