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

package com.intel.analytics.bigdl.dataset.image

import java.awt.color.ColorSpace
import java.nio.file.{Files, Path}

import org.apache.log4j.Logger

object LocalImageFiles {
  Class.forName("javax.imageio.ImageIO")
  Class.forName("java.awt.color.ICC_ColorSpace")
  // Class.forName("sun.java2d.cmm.lcms.LCMS")
  ColorSpace.getInstance(ColorSpace.CS_sRGB).toRGB(Array[Float](0, 0, 0))

  val logger = Logger.getLogger(getClass)

  /**
   * read the folder names, which are the class names, sort the name and convert to an integer
   *
   * @param path
   * @return
   */
  private[bigdl] def readLabels(path: Path): Map[String, Int] = {
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
  private def readPathsWithLabel(path: Path): Array[LocalLabeledImagePath] = {
    val directoryStream = Files.newDirectoryStream(path)
    println(s"Start to read directories $path")
    val labelMap = readLabels(path)
    import scala.collection.JavaConverters._
    directoryStream.asScala.flatMap(dir => {
      println(s"Find class ${dir.getFileName} -> ${labelMap(dir.getFileName.toString)}")
      Files.newDirectoryStream(dir).asScala.map(p =>
        LocalLabeledImagePath(labelMap(dir.getFileName.toString).toFloat, p)).toSeq
    }).toArray.sortWith(
      _.path.getFileName.toString < _.path.getFileName.toString
    )
  }

  /**
   * Read all data file paths into one array.
   *
   * @param path
   * @return
   */
  private def readPathsNoLabel(path: Path): Array[LocalLabeledImagePath] = {
    println(s"Start to read directories $path")
    import scala.collection.JavaConverters._
    Files.newDirectoryStream(path).asScala.map(p =>
      LocalLabeledImagePath(-1, p)).toArray
  }

  /**
   * Read all data file paths into one array.
   * if has label, each subfolder is associate with its label.
   * if no label, give -1 as its label
   *
   * @param path
   * @param hasLabel
   * @return
   */
  private[bigdl] def readPaths(path: Path, hasLabel: Boolean = true)
  : Array[LocalLabeledImagePath] = {
    if (hasLabel) readPathsWithLabel(path) else readPathsNoLabel(path)
  }
}
