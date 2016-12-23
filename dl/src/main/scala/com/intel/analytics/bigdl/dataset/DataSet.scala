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

import com.intel.analytics.bigdl.dataset.image.LocalImageFiles._
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.utils.RandomGenerator
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{SequenceFile, Text}
import org.apache.hadoop.io.SequenceFile.Reader
import org.apache.log4j.Logger
import org.apache.spark.{Partition, SparkContext}
import org.apache.spark.rdd.RDD

import scala.reflect._

/**
 * Represent a set of data, which can be used for training or validation. Data can be shuffled
 *
 * @tparam DataSequence Represent a sequence of data
 */
trait DataSet[DataSequence] {
  /**
   * Get a sequence of data
   *
   * @param looped if the data is looped
   * @return
   */
  def data(looped: Boolean): DataSequence

  /**
   * Change the sequence of data flow from the data set
   */
  def shuffle(): Unit

  /**
   * Return the total size of the data set
   *
   * @return
   */
  def size(): Long
}

/**
 * Mange some 'local' data set, e.g. data in files or memory. We use iterator to access the data
 *
 * @tparam T
 */
trait LocalDataSet[T] extends DataSet[Iterator[T]] {
  def transform[C](transformer: Transformer[T, C]): LocalDataSet[C] = {
    val preDataSource = this
    new LocalDataSet[C] {
      override def shuffle(): Unit = preDataSource.shuffle

      override def size(): Long = preDataSource.size()

      override def data(looped: Boolean): Iterator[C] = transformer(preDataSource.data(looped))
    }
  }

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  /**
   * This operator transform one type of data set to another
   *
   * @param transformer
   * @tparam C
   * @return
   */
  def -> [C](transformer: Transformer[T, C]): LocalDataSet[C] = {
    this.transform(transformer)
  }

  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName
}

/**
 * Represent a set of data cached in an array
 *
 * @tparam T
 */
class LocalArrayDataSet[T] private[dataset](buffer: Array[T]) extends LocalDataSet[T] {
  override def shuffle(): Unit = {
    RandomGenerator.shuffle(buffer)
  }

  override def data(looped: Boolean): Iterator[T] = {
    new Iterator[T] {
      private val index = new AtomicInteger()

      override def hasNext: Boolean = {
        if (looped) {
          true
        } else {
          index.get() < buffer.length
        }
      }

      override def next(): T = {
        val curIndex = index.getAndIncrement()
        if (looped || curIndex < buffer.length) {
          buffer(if (looped) (curIndex % buffer.length) else curIndex)
        } else {
          null.asInstanceOf[T]
        }
      }
    }
  }

  override def size(): Long = buffer.length
}

/**
 * A RDD data set
 *
 * @tparam T
 */
trait DistributedDataSet[T] extends DataSet[RDD[T]] {

  def transform[C: ClassTag](transformer: Transformer[T, C]): DistributedDataSet[C] = {
    val preDataSource = this

    val broadcast = this.originRDD().sparkContext.broadcast(transformer)

    val cachedTransformer =
      preDataSource.originRDD().mapPartitions(_ => Iterator
        .single(broadcast.value.cloneTransformer())
      ).setName("Cached Transformer").persist()
    cachedTransformer.count()

    new DistributedDataSet[C] {
      override def size(): Long = preDataSource.size()

      override def shuffle(): Unit = preDataSource.shuffle()

      override def data(looped: Boolean): RDD[C] =
        preDataSource.data(looped).zipPartitions(cachedTransformer)(
          (data, tran) => tran.next()(data))

      override def originRDD(): RDD[_] = preDataSource.originRDD()
    }
  }

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> [C: ClassTag](transformer: Transformer[T, C]): DistributedDataSet[C] = {
    this.transform(transformer)
  }

  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName

  /**
   * Get the 'origin' RDD of the dataset.
   *
   * @return
   */
  def originRDD(): RDD[_]
}

class CachedDistriDataSet[T: ClassTag] private[dataset] (buffer: RDD[Array[T]])
  extends DistributedDataSet[T] {

  protected lazy val count: Long = buffer.mapPartitions(iter => {
    require(iter.hasNext)
    val array = iter.next()
    require(!iter.hasNext)
    Iterator.single(array.length)
  }).reduce(_ + _)

  protected var indexes: RDD[Array[Int]] = buffer.mapPartitions(iter => {
    Iterator.single(RandomGenerator.shuffle((0 until iter.next().length).toArray))
  }).setName("shuffled index").cache()

  override def data(looped: Boolean): RDD[T] = {
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
        private val _offset = new AtomicInteger(offset)

        override def hasNext: Boolean = {
          if (_looped) true else _offset.get() < localData.length
        }

        override def next(): T = {
          val i = _offset.getAndIncrement()
          if (_looped) {
            localData(indexes(i % localData.length))
          } else {
            if (i < localData.length) {
              localData(indexes(i))
            } else {
              null.asInstanceOf[T]
            }
          }
        }
      }
    })
  }

  override def size(): Long = count

  override def shuffle(): Unit = {
    indexes.unpersist()
    indexes = buffer.mapPartitions(iter => {
      Iterator.single(RandomGenerator.shuffle((0 until iter.next().length).toArray))
    }).setName("shuffled index").cache()
  }

  override def originRDD(): RDD[_] = buffer
}

/**
 * Common used data set generator
 */
object DataSet {
  val logger = Logger.getLogger(getClass)

  /**
   * Generate data set from an array
   */
  def array[T](data: Array[T]): LocalArrayDataSet[T] = {
    new LocalArrayDataSet[T](data)
  }

  def array[T: ClassTag](localData: Array[T], sc: SparkContext, partitionNum: Int)
  : DistributedDataSet[T] = {
    new CachedDistriDataSet[T](
      sc.parallelize(localData, partitionNum)
        // Keep this line, or the array will be send to worker every time
        .coalesce(partitionNum, true)
        .mapPartitions(iter => {
          Iterator.single(iter.toArray)
        }).setName("cached dataset")
        .cache()
    )
  }

  def rdd[T: ClassTag](data: RDD[T], partitionNum: Int):
  DistributedDataSet[T] = {
    new CachedDistriDataSet[T](
      data.coalesce(partitionNum, true)
        .mapPartitions(iter => {
          Iterator.single(iter.toArray)
        }).setName("cached dataset")
        .cache()
    )
  }

  object ImageFolder {
    def paths(path: Path): LocalDataSet[LabeledImageLocalPath] = {
      val buffer = LocalImageFiles.readPaths(path)
      new LocalArrayDataSet[LabeledImageLocalPath](buffer)
    }

    def images(path: Path, scaleTo: Int): LocalDataSet[LabeledBGRImage] = {
      val paths = LocalImageFiles.readPaths(path)
      val total = paths.length
      var count = 1
      val buffer = paths.map(imageFile => {
        logger.info(s"Cache image $count/$total")
        count += 1
        Sample(BGRImage.readImage(imageFile.path, scaleTo), imageFile.label)
      })
      new LocalArrayDataSet[Sample](buffer) -> SampleToBGRImg()
    }

    def images(path: Path, sc: SparkContext, partitionNum: Int, scaleTo: Int)
    : DistributedDataSet[LabeledBGRImage] = {
      val paths = LocalImageFiles.readPaths(path)
      val buffer: Array[Sample] = {
        paths.map(imageFile => {
          Sample(BGRImage.readImage(imageFile.path, scaleTo), imageFile.label)
        })
      }
      array(buffer, sc, partitionNum) -> SampleToBGRImg()
    }
  }

  object SequenceFolder {
    val logger = Logger.getLogger(getClass)
    def paths(path: Path, totalSize: Long): LocalDataSet[SeqFileLocalPath] = {
      logger.info(s"Read sequence files folder $path")
      val buffer: Array[SeqFileLocalPath] = SequenceFiles.findFiles(path)
      logger.info(s"Find ${buffer.length} sequence files")
      require(buffer.length > 0, s"Can't find any sequence files under $path")
      new LocalArrayDataSet[SeqFileLocalPath](buffer) {
        override def size(): Long = {
          totalSize
        }
      }
    }

    def files(url: String, sc: SparkContext, classNum: Int,
      partitionNum: Int): DistributedDataSet[Sample] = {
      val rawData = sc.sequenceFile(url, classOf[Text], classOf[Text]).map(image => {
        Sample(image._2.copyBytes(), image._1.toString.toFloat)
      }).filter(_.label < classNum)

      rdd[Sample](rawData, partitionNum)
    }
  }

}



