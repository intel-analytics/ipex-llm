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

package com.intel.analytics.bigdl.dataset

import java.awt.color.ColorSpace
import java.nio.ByteBuffer
import java.nio.file.{Files, Path, Paths}
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.utils.RandomGenerator
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{SequenceFile, Text}
import org.apache.hadoop.io.SequenceFile.Reader
import org.apache.spark.{Partition, SparkContext}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Represent a set of data, which can be used for training or validation
 *
 * @tparam D A way to access the data
 */
trait DataSet[D] {
  /**
   * Get a method to access the data, e.g. iterator, rdd
   *
   * @return
   */
  def data(): D

  /**
   * Reset a drained data set. For a looped data set, it will do nothing
   */
  def reset(): Unit

  /**
   * Change the data flow from the data set
   */
  def shuffle(): Unit

  /**
   * Return the total size of the data set
   *
   * @return
   */
  def size(): Long

  /**
   * Check whether all data in the data set have been processed. For a looped data set, it will
   * always be false.
   *
   * @return
   */
  def finished(): Boolean
}

/**
 * Mange some 'local' data set, e.g. data in files or memory
 *
 * @tparam T
 */
trait LocalDataSet[T] extends DataSet[Iterator[T]] {
  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  /**
   * This operator transform one type of data set to another
   *
   * @param transformer
   * @tparam C
   * @return
   */
  def ->[C](transformer: Transformer[T, C]): LocalDataSet[C] = {
    val preDataSource = this
    new LocalDataSet[C] {
      override def reset(): Unit = preDataSource.reset

      override def shuffle(): Unit = preDataSource.shuffle

      override def size(): Long = preDataSource.size()

      override def data(): Iterator[C] = transformer.apply(preDataSource.data())

      override def finished(): Boolean = preDataSource.finished()
    }
  }
  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName
}

trait RDDDataSet[T] extends DataSet[RDD[T]] {
  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def ->[C: ClassTag](transformer: Transformer[T, C]): RDDDataSet[C] = {
    val preDataSource = this
    val _transformer = transformer
    new RDDDataSet[C] {
      override def size(): Long = preDataSource.size()

      override def reset(): Unit = preDataSource.reset()

      override def shuffle(): Unit = preDataSource.shuffle()

      override def data(): RDD[C] = preDataSource.data().mapPartitions(_transformer(_))

      override def finished(): Boolean = preDataSource.finished()

      override def partitions(): RDD[_] = preDataSource.partitions()
    }
  }
  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName

  def partitions(): RDD[_]
}

class CachedRDDDataSet[T: ClassTag](buffer: RDD[Array[T]], looped: Boolean)
  extends RDDDataSet[T] {

  protected val count: Long = buffer.mapPartitions(iter => {
    require(iter.hasNext)
    val array = iter.next()
    require(!iter.hasNext)
    Iterator.single(array.length)
  }).reduce(_ + _)

  protected var indexes: RDD[Array[Int]] = buffer.mapPartitions(iter => {
    Iterator.single(RandomGenerator.shuffle((1 to iter.next().length).toArray))
  }).setName("shuffled index").cache()

  override def data(): RDD[T] = {
    val _looped = looped
    buffer.zipPartitions(indexes)((dataIter, indexIter) => {
      val indexes = indexIter.next()
      val localData = dataIter.next()
      val offset = if (_looped) {
        RandomGenerator.RNG.uniform(0, localData.length).toInt
      } else {
        0
      }
      new Iterator[T] {
        private var _offset = new AtomicInteger(offset)

        override def hasNext: Boolean = {
          if (_looped) true else _offset.get() < localData.length
        }

        override def next(): T = {
          localData(indexes(_offset.getAndIncrement()))
        }
      }
    })
  }

  override def finished(): Boolean = false

  override def size(): Long = count

  override def reset(): Unit = {}

  override def shuffle(): Unit = {
    indexes.unpersist()
    indexes = buffer.mapPartitions(iter => {
      Iterator.single(RandomGenerator.shuffle((1 to iter.next().length).toArray))
    }).setName("shuffled index").cache()
    this
  }

  override def partitions(): RDD[_] = buffer
}

object CachedRDDDataSet {
  def apply[T: ClassTag](localData: Array[T], sc: SparkContext, partitionNum: Int,
    looped: Boolean): RDDDataSet[T] = {
    new CachedRDDDataSet[T](
      sc.parallelize(localData, partitionNum)
        .coalesce(partitionNum, true)
        .mapPartitions(iter => {
          Iterator.single(iter.toArray)
        }).setName("cached dataset")
        .cache(),
      looped
    )
  }

  def apply[T: ClassTag](data: RDD[T], partitionNum: Int, looped: Boolean): RDDDataSet[T] = {
    new CachedRDDDataSet[T](
      data.coalesce(partitionNum, true)
        .mapPartitions(iter => {
          Iterator.single(iter.toArray)
        }).setName("cached dataset")
        .cache(),
      looped
    )
  }
}

/**
 * Represent a set of data cached in an array
 *
 * @param looped
 * @tparam T
 */
class ArrayDataSet[T](buffer: Array[T], looped: Boolean) extends LocalDataSet[T] {
  protected val index = new AtomicInteger()

  private val iterator: Iterator[T] = new Iterator[T] {
    override def hasNext: Boolean = finished()

    override def next(): T = {
      val curIndex = index.getAndIncrement()
      buffer(if (looped) (curIndex % buffer.length) else curIndex)
    }
  }

  override def shuffle(): Unit = {
    RandomGenerator.shuffle(buffer)
  }

  override def reset(): Unit = {
    if (!looped) {
      index.set(0)
    }
  }

  override def data(): Iterator[T] = iterator

  override def finished(): Boolean = {
    if (looped) {
      true
    } else {
      index.get() < buffer.length
    }
  }

  override def size(): Long = buffer.length
}

/**
 * This data set will read MNIST files and cache all images in the memory. The MNIST file can be
 * download from http://yann.lecun.com/exdb/mnist/
 */
object MNIST {
  private def load(featureFile: Path, labelFile: Path): Array[(Float, Array[Byte])] = {
    val labelBuffer = ByteBuffer.wrap(Files.readAllBytes(labelFile))
    val featureBuffer = ByteBuffer.wrap(Files.readAllBytes(featureFile))
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

  def LocalDataSet(imagesFile: Path, labelsFile: Path, looped: Boolean)
  : LocalDataSet[(Float, Array[Byte])] = {
    val buffer = load(imagesFile, labelsFile)
    new ArrayDataSet[(Float, Array[Byte])](buffer, looped)
  }

  def RDDDataSet(imagesFile: Path, labelsFile: Path, looped: Boolean, sc: SparkContext,
    partitionNum: Int): RDDDataSet[(Float, Array[Byte])] = {
    val buffer = load(imagesFile, labelsFile)
    CachedRDDDataSet(buffer, sc, partitionNum, looped)
  }
}

/**
 * This dat set will read images in Cifar dataset and cache them in the memory. The Cifar dataset
 * can be download from https://www.cs.toronto.edu/~kriz/cifar.html
 */
object Cifar {

  import DirectoryAsLabel._

  Class.forName("javax.imageio.ImageIO")
  Class.forName("java.awt.color.ICC_ColorSpace")
  Class.forName("sun.java2d.cmm.lcms.LCMS")
  ColorSpace.getInstance(ColorSpace.CS_sRGB).toRGB(Array[Float](0, 0, 0))

  def LocalDataSet(path: Path, looped: Boolean, scaleTo: Int = 32)
  : LocalDataSet[(Float, Array[Byte])] = {
    val paths = readPaths(path)
    val buffer: Array[(Float, Array[Byte])] = {
      paths.map(imageFile => {
        (imageFile._1, RGBImage.readImage(imageFile._2, scaleTo))
      })
    }
    new ArrayDataSet[(Float, Array[Byte])](buffer, looped)
  }

  def RDDDataSet(path: Path, looped: Boolean, scaleTo: Int = 32, sc: SparkContext,
    partitionNum: Int): RDDDataSet[(Float, Array[Byte])] = {
    val paths = readPaths(path)
    val buffer: Array[(Float, Array[Byte])] = {
      paths.map(imageFile => {
        (imageFile._1, RGBImage.readImage(imageFile._2, scaleTo))
      })
    }
    CachedRDDDataSet(buffer, sc, partitionNum, looped)
  }
}

/**
 * ImageNet2012 DataSet
 */
object ImageNet {
  /**
   * This dataset will manage imagenet2012 files. You can download the data
   * from http://www.image-net.org/
   *
   * Notice it won't cache the data but only the file paths in the memory
   *
   * @param path
   * @param looped
   */
  def PathDataSet(path: Path, looped: Boolean): LocalDataSet[(Float, Path)] = {
    import DirectoryAsLabel._
    val buffer = readPaths(path)
    new ArrayDataSet[(Float, Path)](buffer, looped)
  }

  /**
   * This dataset will manage imagenet2012 in Hadoop sequence file format store on your local disk.
   *
   * You can generate the sequence file by using ImageNetSeqFileGenerator.
   *
   * @param path
   * @param totalSize
   * @param looped
   */
  def LocalSeqDataSet(path: Path, totalSize: Long, looped: Boolean): LocalDataSet[(Path)] = {
    val buffer: Array[Path] = findFiles(path)
    new ArrayDataSet[Path](buffer, looped) {
      override def size(): Long = {
        totalSize
      }
    }
  }

  private def findFiles(path : Path) : Array[Path] = {
    val directoryStream = Files.newDirectoryStream(path)
    import scala.collection.JavaConverters._
    directoryStream.asScala.map(_.toAbsolutePath.toString).map(Paths.get(_)).toArray
  }

  def RDDSeqDataSet(url: String, sc: SparkContext, classNum: Int, looped: Boolean,
    partitionNum: Int): RDDDataSet[(Float, Array[Byte])] = {
    val rawData = sc.sequenceFile(url, classOf[Text], classOf[Text]).map(image => {
      (image._1.toString.toFloat, image._2.copyBytes())
    }).filter(_._1 < classNum)

    CachedRDDDataSet[(Float, Array[Byte])](rawData, partitionNum, looped)
  }
}

/**
 * Many dataset seperates data files into different folders by the class. These are helper
 * functions process such structure.
 */
object DirectoryAsLabel {
  /**
   * read the folder names, which are the class names, sort the name and convert to an integer
   *
   * @param path
   * @return
   */
  def readLabels(path: Path): Map[String, Int] = {
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
  def readPaths(path: Path): Array[(Float, Path)] = {
    val directoryStream = Files.newDirectoryStream(path)
    println(s"Start to read directories $path")
    val labelMap = readLabels(path)
    import scala.collection.JavaConverters._
    directoryStream.asScala.flatMap(dir => {
      println(s"Find class ${dir.getFileName} -> ${labelMap(dir.getFileName.toString)}")
      Files.newDirectoryStream(dir).asScala.map(p =>
        (labelMap(dir.getFileName.toString).toFloat, p)).toSeq
    }).toArray.sortWith(
      _._2.getFileName.toString < _._2.getFileName.toString
    )
  }
}

