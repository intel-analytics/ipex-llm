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

import java.io.File

import java.nio.ByteBuffer

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.DataSet.SeqFileFolder.readLabel
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.zoo.common.Utils
import com.intel.analytics.zoo.feature.common.Preprocessing
import org.apache.commons.io.FileUtils
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.opencv.imgcodecs.Imgcodecs

import scala.reflect.ClassTag
import scala.collection.JavaConverters._
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.hadoop.fs.Path

/**
 * ImageSet wraps a set of ImageFeature
 */
abstract class ImageSet {

  val labelMap: Option[Map[String, Int]]

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
   * Convert ImageSet to DataSet of Sample.
   */
  def toDataSet[T: ClassTag](): DataSet[Sample[T]]

}

class LocalImageSet(var array: Array[ImageFeature],
                    val labelMap: Option[Map[String, Int]] = None) extends ImageSet {
  override def transform(transformer: Preprocessing[ImageFeature, ImageFeature]): ImageSet = {
    array = transformer.apply(array.toIterator).toArray
    this
  }

  override def isLocal(): Boolean = true

  override def isDistributed(): Boolean = false

  override def toImageFrame(): ImageFrame = {
    ImageFrame.array(array)
  }

  override def toDataSet[T: ClassTag](): DataSet[Sample[T]] = {

    DataSet.array(array.map(_[Sample[T]](ImageFeature.sample)))
  }
}

class DistributedImageSet(var rdd: RDD[ImageFeature],
                          val labelMap: Option[Map[String, Int]] = None) extends ImageSet {
  override def transform(transformer: Preprocessing[ImageFeature, ImageFeature]): ImageSet = {
    rdd = transformer(rdd)
    this
  }

  override def isLocal(): Boolean = false

  override def isDistributed(): Boolean = true

  override def toImageFrame(): ImageFrame = {
    ImageFrame.rdd(rdd)
  }

  override def toDataSet[T: ClassTag](): DataSet[Sample[T]] = {
    DataSet.rdd(rdd.map(_[Sample[T]](ImageFeature.sample)))
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
   * create LocalImageSet
   * @param data array of ImageFeature
   * @param labelMap the labelMap of this ImageSet, a mapping from class names to label
   */
  def array(data: Array[ImageFeature],
            labelMap: Map[String, Int]): LocalImageSet = {
    new LocalImageSet(data, Option(labelMap))
  }

  private def transform(imageSet: ImageSet,
                        resizeH: Int, resizeW: Int, imageCodec: Int): ImageSet = {
    if (resizeW == -1 || resizeH == -1) {
      imageSet -> ImageBytesToMat(imageCodec = imageCodec)
    } else {
      imageSet -> BufferedImageResize(resizeH, resizeW) ->
        ImageBytesToMat(imageCodec = imageCodec)
    }
  }

  /**
   * create LocalImageSet from array of bytes
   * @param data nested array of bytes, expect inner array is a image
   * @param resizeH height after resize, by default is -1 which will not resize the image
   * @param resizeW width after resize, by default is -1 which will not resize the image
   * @param imageCodec specifying the color type of a loaded image, same as in OpenCV.imread.
   *              By default is Imgcodecs.CV_LOAD_IMAGE_UNCHANGED
   */
  def array(data: Array[Array[Byte]], resizeH: Int = -1, resizeW: Int = -1,
            imageCodec: Int = Imgcodecs.CV_LOAD_IMAGE_UNCHANGED,
            labelMap: Map[String, Int] = null): ImageSet = {
    val images = data.map(ImageFeature(_))
    val imageSet = ImageSet.array(images, labelMap)
    transform(imageSet, resizeH, resizeW, imageCodec)
  }

  /**
   * create DistributedImageSet
   * @param data rdd of ImageFeature
   */
  def rdd(data: RDD[ImageFeature],
          labelMap: Map[String, Int] = null): DistributedImageSet = {

    new DistributedImageSet(data, Option(labelMap))
  }

  /**
   * create DistributedImageSet for a RDD of array bytes
   * @param data rdd of array of bytes
   * @param resizeH height after resize, by default is -1 which will not resize the image
   * @param resizeW width after resize, by default is -1 which will not resize the image
   * @param imageCodec specifying the color type of a loaded image, same as in OpenCV.imread.
   *              By default is Imgcodecs.CV_LOAD_IMAGE_UNCHANGED
   */
  def rddBytes(data: RDD[Array[Byte]], resizeH: Int = -1, resizeW: Int = -1,
               imageCodec: Int = Imgcodecs.CV_LOAD_IMAGE_UNCHANGED,
               labelMap: Map[String, Int] = null): ImageSet = {
    val images = data.map(ImageFeature(_))
    val imageSet = ImageSet.rdd(images, labelMap)
    transform(imageSet, resizeH, resizeW, imageCodec)
  }

  /**
   * Read images as Image Set
   * if sc is defined, Read image as DistributedImageSet from local file system or HDFS
   * if sc is null, Read image as LocalImageSet from local file system
   *
   * @param path path to read images
   * if sc is defined, path can be local or HDFS. Wildcard character are supported.
   * if sc is null, path is local directory/image file/image file with wildcard character
   *
   * if withLabel is set to true, path should be a directory that have two levels. The
   * first level is class folders, and the second is images. All images belong to a same
   * class should be put into the same class folder. So each image in the path is labeled by the
   * folder it belongs.
   *
   * @param sc SparkContext
   * @param minPartitions A suggestion value of the minimal partition number
   * @param resizeH height after resize, by default is -1 which will not resize the image
   * @param resizeW width after resize, by default is -1 which will not resize the image
   * @param imageCodec specifying the color type of a loaded image, same as in OpenCV.imread.
   *              By default is Imgcodecs.CV_LOAD_IMAGE_UNCHANGED
   * @param withLabel whether to treat folders in the path as image classification labels and read
   *                  the labels into ImageSet.
   * @return ImageSet
   */
  def read(path: String, sc: SparkContext = null, minPartitions: Int = 1,
           resizeH: Int = -1, resizeW: Int = -1,
           imageCodec: Int = Imgcodecs.CV_LOAD_IMAGE_UNCHANGED,
           withLabel: Boolean = false, oneBasedLabel: Boolean = true): ImageSet = {
    val imageSet = if (null != sc) {
      readToDistributedImageSet(path, minPartitions, sc, withLabel, oneBasedLabel)
    } else {
      readToLocalImageSet(path, withLabel, oneBasedLabel)
    }
    transform(imageSet, resizeH, resizeW, imageCodec)
  }

  private def readToDistributedImageSet(pathStr: String,
                                        minPartitions: Int,
                                        sc: SparkContext,
                                        withLabel: Boolean,
                                        oneBasedLabel: Boolean): ImageSet = {
    if (withLabel) {
      val path = new Path(pathStr)

      val fsSys = path.getFileSystem(sc.hadoopConfiguration)
      val dirPath = fsSys.getFileStatus(path).getPath.toUri.getRawPath
      val classFolders = fsSys.listStatus(path).filter(_.isDirectory)
      val newPathsString = classFolders.map(_.getPath.toUri.toString).mkString(",")
      val labelMap = classFolders.map(_.getPath.getName)
        .sorted.zipWithIndex.map { c =>
        if (oneBasedLabel) c._1 -> (c._2 + 1)
        else c._1 -> c._2
      }.toMap
      val images = sc.binaryFiles(newPathsString, minPartitions).map { case (p, stream) =>
        val rawFilePath = new Path(p).toUri.getRawPath
        assert(rawFilePath.startsWith(dirPath),
          s"directory path: $dirPath does not match file path $rawFilePath")
        val classStr = rawFilePath
          .substring(dirPath.length + 1).split(File.separator)(0)
        val label = labelMap(classStr)
        ImageFeature(stream.toArray(), uri = p, label = Tensor[Float](T(label)))
      }

      ImageSet.rdd(images, labelMap)
    } else {
      val images = sc.binaryFiles(pathStr, minPartitions).map { case (p, stream) =>
        ImageFeature(stream.toArray(), uri = p)
      }
      ImageSet.rdd(images)
    }
  }

  private def readToLocalImageSet(pathStr: String,
                                  withLabel: Boolean,
                                  oneBasedLabel: Boolean): ImageSet = {

    val basePath = Paths.get(pathStr)

    if (withLabel) {
      val classFolders = Files.newDirectoryStream(basePath).asScala
        .filter(_.toFile.isDirectory)

      val labelMap = classFolders
        .map(_.getFileName.toString)
        .toArray.sortWith(_ < _).zipWithIndex.map{ c =>
        if (oneBasedLabel) c._1 -> (c._2 + 1)
        else c._1 -> c._2
      }.toMap

      val files = classFolders.flatMap { p =>
        Utils.listLocalFiles(p.toAbsolutePath.toString)
      }

      val images = files.map { p =>
        val classStr = p.getAbsolutePath
          .substring(basePath.toFile.getAbsolutePath.length + 1).split(File.separator)(0)
        val label = labelMap(classStr)
        ImageFeature(FileUtils.readFileToByteArray(p),
          uri = p.getAbsolutePath, label = Tensor[Float](T(label)))
      }.toArray

      ImageSet.array(images, labelMap)
    } else {
      val files = Utils.listLocalFiles(pathStr)
      val images = files.map { p =>
        ImageFeature(FileUtils.readFileToByteArray(p), uri = p.getAbsolutePath)
      }
      ImageSet.array(images)
    }

  }

  /**
   * Read images (with labels) from Hadoop SequenceFiles as ImageSet.
   *
   * @param path The folder path that contains sequence files.
   *             Local or distributed file system (such as HDFS) are supported.
   * @param sc An instance of SparkContext.
   * @param minPartitions Integer. A suggestion value of the minimal partition number for input
   *                      texts. Default is 1.
   * @param classNum Integer. The number of image categories. Default is 1000 for ImageNet.
   * @return DistributedImageSet.
   */
  def readSequenceFiles(path: String, sc: SparkContext,
                        minPartitions: Int = 1, classNum: Int = 1000): DistributedImageSet = {
    val images = sc.sequenceFile(path, classOf[Text], classOf[Text], minPartitions).map(image => {
      val rawBytes = image._2.copyBytes()
      val label = Tensor[Float](T(readLabel(image._1).toFloat))
      val imgBuffer = ByteBuffer.wrap(rawBytes)
      val width = imgBuffer.getInt
      val height = imgBuffer.getInt
      val bytes = new Array[Byte](3 * width * height)
      System.arraycopy(imgBuffer.array(), 8, bytes, 0, bytes.length)
      val imf = ImageFeature(bytes, label)
      imf(ImageFeature.originalSize) = (height, width, 3)
      imf
    }).filter(_[Tensor[Float]](ImageFeature.label).valueAt(1) <= classNum)
    val imageSet = ImageSet.rdd(images)
    (imageSet -> ImagePixelBytesToMat()).toDistributed()
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
