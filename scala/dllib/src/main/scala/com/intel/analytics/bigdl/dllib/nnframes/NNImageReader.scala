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

package com.intel.analytics.bigdl.dllib.nnframes

import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.dllib.feature.image.ImageSet
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.opencv.core.{CvType, Mat}
import org.opencv.imgcodecs.Imgcodecs

import scala.language.existentials

/**
 * Definition for image data in a DataFrame
 */
object NNImageSchema {

  /**
   * Schema for the image column in a DataFrame. Image data is saved in an array of Bytes.
   */
  val byteSchema = StructType(
    StructField("origin", StringType, true) ::
      StructField("height", IntegerType, false) ::
      StructField("width", IntegerType, false) ::
      StructField("nChannels", IntegerType, false) ::
      // OpenCV-compatible type: CV_8UC3, CV_8UC1 in most cases
      StructField("mode", IntegerType, false) ::
      // Bytes in row-wise BGR
      StructField("data", BinaryType, false) :: Nil)

  /**
   * Schema for the image column in a DataFrame. Image data is saved in an array of Floats.
   */
  val floatSchema = StructType(
    StructField("origin", StringType, true) ::
      StructField("height", IntegerType, false) ::
      StructField("width", IntegerType, false) ::
      StructField("nChannels", IntegerType, false) ::
      // OpenCV-compatible type: CV_32FC3 in most cases
      StructField("mode", IntegerType, false) ::
      // floats in OpenCV-compatible order: row-wise BGR in most cases
      StructField("data", new ArrayType(FloatType, false), false) :: Nil)

  private[bigdl] def imf2Row(imf: ImageFeature): Row = {
      val (mode, data) = if (imf.contains(ImageFeature.imageTensor)) {
      val floatData = imf(ImageFeature.imageTensor).asInstanceOf[Tensor[Float]].storage().array()
      val cvType = imf.getChannel() match {
        case 1 => CvType.CV_32FC1
        case 3 => CvType.CV_32FC3
        case 4 => CvType.CV_32FC4
        case other => throw new IllegalArgumentException(s"Unsupported number of channels:" +
          s" $other in ${imf.uri()}. Only 1, 3 and 4 are supported.")
      }
      (cvType, floatData)
    } else if (imf.contains(ImageFeature.mat)) {
      val mat = imf.opencvMat()
      val cvType = mat.`type`()
      val bytesData = OpenCVMat.toBytePixels(mat)._1
      (cvType, bytesData)
    } else {
      throw new IllegalArgumentException(s"ImageFeature should have imageTensor or mat.")
    }

    Row(
      imf.uri(),
      imf.getHeight(),
      imf.getWidth(),
      imf.getChannel(),
      mode,
      data
    )
  }

  private[bigdl] def row2IMF(row: Row): ImageFeature = {
    val (origin, h, w, c) = (row.getString(0), row.getInt(1), row.getInt(2), row.getInt(3))
    val imf = ImageFeature()
    imf.update(ImageFeature.uri, origin)
    imf.update(ImageFeature.size, (h, w, c))
    val storageType = row.getInt(4)
    storageType match {
      case CvType.CV_8UC3 | CvType.CV_8UC1 | CvType.CV_8UC4 =>
        val bytesData = row.getAs[Array[Byte]](5)
        val opencvMat = OpenCVMat.fromPixelsBytes(bytesData, h, w, c)
        imf(ImageFeature.mat) = opencvMat
        imf(ImageFeature.originalSize) = opencvMat.shape()
      case CvType.CV_32FC3 | CvType.CV_32FC1 | CvType.CV_32FC4 =>
        val data = row.getSeq[Float](5).toArray
        val size = Array(h, w, c)
        val ten = Tensor(Storage(data)).resize(size)
        imf.update(ImageFeature.imageTensor, ten)
      case _ =>
        throw new IllegalArgumentException(s"Unsupported data type in imageColumn: $storageType")
    }
    imf
  }

  /**
   * Gets the origin of the image
   *
   * @return The origin of the image
   */
  def getOrigin(row: Row): String = row.getString(0)

  /**
   * Extracts the origin info from Image column and add it to a new column.
   *
   * @param imageDF input DataFrame that contains an image column
   * @param imageColumn image column name
   * @param originColumn name of new column
   * @return The origin of the image
   */
  def withOriginColumn(
      imageDF: DataFrame,
      imageColumn: String = "image",
      originColumn: String = "origin"): DataFrame = {
      val getPathUDF = udf { row: Row => getOrigin(row) }
    imageDF.withColumn(originColumn, getPathUDF(col(imageColumn)))
  }

}

/**
 * Primary DataFrame-based image loading interface, defining API to read images into DataFrame.
 */
object NNImageReader {

  /**
   * DataFrame with a single column of images named "image" (nullable)
   */
  private val imageColumnSchema =
    StructType(StructField("image", NNImageSchema.byteSchema, true) :: Nil)

  /**
   * Read the directory of images into DataFrame from the local or remote source.
   *
   * @param path Directory to the input data files, the path can be comma separated paths as the
   *             list of inputs. Wildcards path are supported similarly to sc.binaryFiles(path).
   * @param sc SparkContext to be used.
   * @param minPartitions Number of the DataFrame partitions,
   *                      if omitted uses defaultParallelism instead
   * @param resizeH height after resize, by default is -1 which will not resize the image
   * @param resizeW width after resize, by default is -1 which will not resize the image
   * @param imageCodec specifying the color type of a loaded image, same as in OpenCV.imread.
   *              By default is Imgcodecs.CV_LOAD_IMAGE_UNCHANGED.
   *              >0 Return a 3-channel color image. Note In the current implementation the
   *                 alpha channel, if any, is stripped from the output image. Use negative value
   *                  if you need the alpha channel.
   *              =0 Return a grayscale image.
   *              <0 Return the loaded image as is (with alpha channel if any).
   * @return DataFrame with a single column "image" of images;
   *         see DLImageSchema.byteSchema for the details
   */
  def readImages(path: String, sc: SparkContext, minPartitions: Int = 1,
                 resizeH: Int = -1, resizeW: Int = -1,
                 imageCodec: Int = Imgcodecs.CV_LOAD_IMAGE_UNCHANGED): DataFrame = {
    val imageSet = ImageSet.read(path, sc, minPartitions, resizeH, resizeW, imageCodec)
    val rowRDD = imageSet.toDistributed().rdd.map { imf =>
      Row(NNImageSchema.imf2Row(imf))
    }
    SQLContext.getOrCreate(sc).createDataFrame(rowRDD, imageColumnSchema)
  }

}
