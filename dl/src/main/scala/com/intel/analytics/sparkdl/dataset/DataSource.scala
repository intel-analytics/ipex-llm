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

package com.intel.analytics.sparkdl.dataset

import java.awt.color.ColorSpace
import java.nio.ByteBuffer
import java.nio.file.{Files, Path, Paths}
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.sparkdl.utils.RandomGenerator
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{SequenceFile, Text}
import org.apache.hadoop.io.SequenceFile.Reader
import org.apache.spark.rdd.RDD

import scala.collection.Iterator
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

trait DataSource[T] extends Iterator[T] {
  def reset(): Unit

  def shuffle(): Unit

  def total(): Long
}

trait LocalDataSource[T] extends DataSource[T] {
  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> [C](transformer: Transformer[T, C]): LocalDataSource[C] = {
    val preDataSource = this
    new LocalDataSource[C] {
      private val iterator = transformer.transform(preDataSource)

      override def reset(): Unit = preDataSource.reset

      override def shuffle(): Unit = preDataSource.shuffle

      override def next(): C = iterator.next

      override def hasNext: Boolean = iterator.hasNext

      override def total(): Long = preDataSource.total()
    }
  }
  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName
}

trait RDDDataSource[T] extends DataSource[RDD[T]] {
  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> [C: ClassTag](transformer: Transformer[T, C]): RDDDataSource[C] = {
    val preDataSource = this
    val _transformer = transformer
    new RDDDataSource[C] {
      override def total(): Long = preDataSource.total()

      override def reset(): Unit = preDataSource.reset()

      override def shuffle(): Unit = preDataSource.shuffle()

      override def next(): RDD[C] = preDataSource.next().mapPartitions(pre => {
        _transformer.transform(pre)
      })

      override def hasNext: Boolean = preDataSource.hasNext
    }
  }
  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName
}

abstract class ArrayDataSource[T](looped: Boolean) extends LocalDataSource[T] {
  protected val index = new AtomicInteger()

  protected val data: Array[T]

  override def shuffle(): Unit = {
    var i = 0
    while (i < data.length) {
      val exchange = i + RandomGenerator.RNG.uniform(0, data.length - i).toInt
      val tmp = data(exchange)
      data(exchange) = data(i)
      data(i) = tmp
      i += 1
    }
  }

  override def reset(): Unit = {
    index.set(0)
  }

  override def next(): T = {
    val curIndex = index.getAndIncrement()
    data(if (looped) (curIndex % data.length) else curIndex)
  }

  override def hasNext: Boolean = {
    if (looped) {
      true
    } else {
      index.get() < data.length
    }
  }

  override def total(): Long = data.length
}

class MNISTDataSource(trainDataPath: String, validationDataPath: String, looped: Boolean)
  extends ArrayDataSource[(Float, Array[Byte])](looped) {

  override val data = load(trainDataPath, validationDataPath)

  private def load(featureFile: String, labelFile: String): Array[(Float, Array[Byte])] = {
    val labelBuffer = ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))
    val featureBuffer = ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    val labelMagicNumber = labelBuffer.getInt()

    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()

    val result = new Array[(Float, Array[Byte])](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum))
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = (labelBuffer.get().toFloat + 1.0f, img)
      i += 1
    }

    result
  }
}

class CifarDataSource(path: Path, looped: Boolean, scaleTo: Int = 32)
  extends ArrayDataSource[(Float, Array[Byte])](looped) with DirectoryAsLabelDataSet {

  private val paths = loadPaths(path)

  override protected val data: Array[(Float, Array[Byte])] = paths.map(imageFile => {
    (imageFile._1, RGBImage.readImage(imageFile._2, scaleTo))
  })
}

object ImageNetDataSource {
  def apply(path: Path, looped: Boolean): ImageNetDataSource = new ImageNetDataSource(path, looped)
}

class ImageNetDataSource(path: Path, looped: Boolean)
  extends ArrayDataSource[(Float, Path)](looped) with DirectoryAsLabelDataSet {

  override val data: Array[(Float, Path)] = loadPaths(path)
}

class ImageNetSeqDataSource(path : Path, totalSize : Long, looped: Boolean)
  extends ArrayDataSource[Path](looped) {
  override protected val data: Array[Path] = findFiles(path)

  override def total() : Long = {
    totalSize
  }

  private def findFiles(path : Path) : Array[Path] = {
    val directoryStream = Files.newDirectoryStream(path)
    import scala.collection.JavaConverters._
    directoryStream.asScala.map(_.toAbsolutePath.toString).map(Paths.get(_)).toArray
  }
}

trait DirectoryAsLabelDataSet {
  def loadPaths(path: Path): Array[(Float, Path)] = {
    Class.forName("javax.imageio.ImageIO")
    Class.forName("java.awt.color.ICC_ColorSpace")
    Class.forName("sun.java2d.cmm.lcms.LCMS")
    ColorSpace.getInstance(ColorSpace.CS_sRGB).toRGB(Array[Float](0, 0, 0))

    val directoryStream = Files.newDirectoryStream(path)
    println(s"Start to read directories $path")
    val labelMap = getLabelMap(path)
    import scala.collection.JavaConverters._
    directoryStream.asScala.flatMap(dir => {
      println(s"Find class ${dir.getFileName} -> ${labelMap(dir.getFileName.toString)}")
      Files.newDirectoryStream(dir).asScala.map(p =>
        (labelMap(dir.getFileName.toString).toFloat, p)).toSeq
    }).toArray.sortWith(
      _._2.getFileName.toString < _._2.getFileName.toString
    )
  }

  def getLabelMap(path: Path): Map[String, Int] = {
    import scala.collection.JavaConverters._
    Files.newDirectoryStream(path).asScala.map(_.getFileName.toString)
      .toArray.sortWith(_ < _).zipWithIndex.map(c => c._1 -> (c._2 + 1)).toMap
  }
}
