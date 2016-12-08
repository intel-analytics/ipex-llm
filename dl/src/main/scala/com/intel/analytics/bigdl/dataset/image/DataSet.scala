/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.dataset.image

import java.awt.color.ColorSpace
import java.nio.file.{Files, Path, Paths}

import com.intel.analytics.bigdl.dataset.{CachedDistriDataSet, _}
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext

object LocalImageFiles {
  Class.forName("javax.imageio.ImageIO")
  Class.forName("java.awt.color.ICC_ColorSpace")
  Class.forName("sun.java2d.cmm.lcms.LCMS")
  ColorSpace.getInstance(ColorSpace.CS_sRGB).toRGB(Array[Float](0, 0, 0))

  // scalastyle:off methodName
  def LocalDataSet(path: Path, looped: Boolean, scaleTo: Int = 32)
  : LocalDataSet[LabeledImageLocalPath] = {
    val buffer = readPaths(path)
    new LocalArrayDataSet[LabeledImageLocalPath](buffer, looped)
  }

  def DistriDataSet(path: Path, looped: Boolean, sc: SparkContext,
    partitionNum: Int, scaleTo: Int = 32): DistributedDataSet[(Float, Array[Byte])] = {
    val paths = readPaths(path)
    val buffer: Array[(Float, Array[Byte])] = {
      paths.map(imageFile => {
        (imageFile.label, RGBImage.readImage(imageFile.path, scaleTo))
      })
    }
    CachedDistriDataSet(buffer, sc, partitionNum, looped)
  }
  // scalastyle:on methodName

  /**
   * read the folder names, which are the class names, sort the name and convert to an integer
   *
   * @param path
   * @return
   */
  private def readLabels(path: Path): Map[String, Int] = {
    import scala.collection.JavaConverters._
    Files.newDirectoryStream(path).asScala.map(_.getFileName.toString)
      .toArray.sortWith(_ < _).zipWithIndex.map(c => c._1 -> (c._2 + 1)).toMap
  }

  /**
   * Read all data file paths into one array. Each path is associate with its label.
   *
   * @param path
   * @return
   */
  private def readPaths(path: Path): Array[LabeledImageLocalPath] = {
    val directoryStream = Files.newDirectoryStream(path)
    println(s"Start to read directories $path")
    val labelMap = readLabels(path)
    import scala.collection.JavaConverters._
    directoryStream.asScala.flatMap(dir => {
      println(s"Find class ${dir.getFileName} -> ${labelMap(dir.getFileName.toString)}")
      Files.newDirectoryStream(dir).asScala.map(p =>
        LabeledImageLocalPath(labelMap(dir.getFileName.toString).toFloat, p)).toSeq
    }).toArray.sortWith(
      _.path.getFileName.toString < _.path.getFileName.toString
    )
  }
}

object SequenceFiles {
  /**
   * This dataset will manage imagenet2012 in Hadoop sequence file format store on your local disk.
   *
   * You can generate the sequence file by using ImageNetSeqFileGenerator.
   *
   * @param path
   * @param totalSize
   * @param looped
   */
  def LocalFiles(path: Path, totalSize: Long, looped: Boolean): LocalDataSet[SeqFileLocalPath] = {
    val buffer: Array[SeqFileLocalPath] = findFiles(path)
    new LocalArrayDataSet[SeqFileLocalPath](buffer, looped) {
      override def size(): Long = {
        totalSize
      }
    }
  }

  private def findFiles(path: Path): Array[SeqFileLocalPath] = {
    val directoryStream = Files.newDirectoryStream(path)
    import scala.collection.JavaConverters._
    directoryStream.asScala.map(_.toAbsolutePath.toString)
      .filter(_.endsWith(".seq")).toArray.sortWith(_ < _).map(p => SeqFileLocalPath(Paths.get(p)))
  }

  def HDFSFiles(url: String, sc: SparkContext, classNum: Int, looped: Boolean,
    partitionNum: Int): DistributedDataSet[(Float, Array[Byte])] = {
    val rawData = sc.sequenceFile(url, classOf[Text], classOf[Text]).map(image => {
      (image._1.toString.toFloat, image._2.copyBytes())
    }).filter(_._1 < classNum)

    CachedDistriDataSet[(Float, Array[Byte])](rawData, partitionNum, looped)
  }
  // scalastyle:on methodName
}
