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

package com.intel.analytics.zoo.feature

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.{AbstractDataSet, DistributedDataSet, Transformer}
import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.zoo.feature.common.{ArrayLike, ArrayLikeWrapper}
import com.intel.analytics.zoo.feature.pmem._
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.reflect.ClassTag

/**
 * A set of data which is used in the model optimization process. The FeatureSet can be access in
 * a random data sample sequence. In the training process, the data sequence is a looped endless
 * sequence. While in the validation process, the data sequence is a limited length sequence.
 * User can use the data() method to get the data sequence.
 *
 * The sequence of the data is not fixed. It can be changed by the shuffle() method.
 *
 * User can create a dataset from a RDD, an array and a folder, etc. The DataSet object provides
 * many factory methods.
 *
 * @tparam D Data type
 * @tparam DataSequence Represent a sequence of data
 */
trait AbstractFeatureSet[D, DataSequence] extends AbstractDataSet[D, DataSequence]{
  /**
   * Get a sequence of data
   *
   * @param train if the data is used in train. If yes, the data sequence is a looped endless
   *              sequence, or it has a limited length.
   * @return data sequence
   */
  def data(train: Boolean): DataSequence

  /**
   * Change the order of the data sequence from the data set
   */
  def shuffle(): Unit

  /**
   * Total size of the data set
   * @return
   */
  def size(): Long

  /**
   * Helper function to transform the data type in the data set.
   * @param transformer
   * @tparam C
   * @return
   */
  override def transform[C: ClassTag](transformer: Transformer[D, C]): FeatureSet[C]

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  /**
   * Helper function to transform the data type in the data set.
   *
   * @param transformer
   * @tparam C
   * @return
   */
  override def -> [C: ClassTag](transformer: Transformer[D, C]): FeatureSet[C] = {
    this.transform(transformer)
  }

  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName

  /**
   * Convert FeatureSet to com.intel.analytics.bigdl.DataSet
   * @return DataSet[D]
   */
  def toDataSet(): DataSet[D]
}

/**
 * Represent a distributed data. Use RDD to go through all data.
 *
 * @tparam T
 */
trait DistributedFeatureSet[T] extends AbstractFeatureSet[T, RDD[T]] {

  override def transform[C: ClassTag](transformer: Transformer[T, C]): DistributedFeatureSet[C] = {
    val preFeatureSet = this

    val broadcast = this.originRDD().sparkContext.broadcast(transformer)

    val cachedTransformer =
      preFeatureSet.originRDD().mapPartitions(_ => Iterator
        .single(broadcast.value.cloneTransformer())
      ).setName("Cached Transformer").persist()

    new DistributedFeatureSet[C] {
      override def size(): Long = preFeatureSet.size()

      override def shuffle(): Unit = preFeatureSet.shuffle()

      override def data(train: Boolean): RDD[C] =
        preFeatureSet.data(train).zipPartitions(cachedTransformer)(
          (data, tran) => tran.next()(data))

      override def originRDD(): RDD[_] = preFeatureSet.originRDD()

      override def cache(): Unit = {
        cachedTransformer.count()
        isCached = true
      }

      override def unpersist(): Unit = {
        preFeatureSet.unpersist()
        cachedTransformer.unpersist()
        isCached = false
      }

      override def toDistributed(): DistributedDataSet[C] = {
        new DistributedDataSetWrapper[C](this)
      }
    }
  }

  /**
   * Get the 'origin' RDD of the dataset.
   *
   * @return
   */
  def originRDD(): RDD[_]

  /**
   * Trigger the computation of this dataset and cache it in memory.
   */
  def cache(): Unit = {
    if (originRDD() != null) {
      originRDD().count()
    }
    isCached = true
  }

  /**
   * Unpersist rdd.
   */
  def unpersist(): Unit = {
    if (originRDD() != null) {
      originRDD().unpersist()
      isCached = false
    }
  }

  /**
   * Check if rdd is cached.
   */
  var isCached = false

  override def toDataSet(): DataSet[T] = {
    toDistributed()
  }
}

/**
 * Wrap a featureSet to DistributedDataSet.
 * @param featureSet
 * @param ev$1
 * @tparam T
 */
private[zoo] class DistributedDataSetWrapper[T: ClassTag](featureSet: DistributedFeatureSet[T])
  extends DistributedDataSet[T]{

  override def data(train: Boolean): RDD[T] = {
    featureSet.data(train)
  }

  override def size(): Long = featureSet.size()

  override def shuffle(): Unit = {
    featureSet.shuffle()
  }

  override def originRDD(): RDD[_] = featureSet.originRDD()

  override def cache(): Unit = {
    featureSet.cache()
  }

  override def unpersist(): Unit = {
    featureSet.unpersist()
  }

}

/**
 * Wrap a RDD as a FeatureSet.
 * @param buffer
 */
// T is the returning value type. like ByteRecord
class CachedDistributedFeatureSet[T: ClassTag]
(buffer: RDD[ArrayLike[T]])
  extends DistributedFeatureSet[T]{

  protected lazy val count: Long = buffer.mapPartitions(iter => {
    require(iter.hasNext)
    val array = iter.next()
    require(!iter.hasNext)
    Iterator.single(array.length)
  }).reduce(_ + _)

  protected var indexes: RDD[Array[Int]] = buffer.mapPartitions(iter => {
    Iterator.single[Array[Int]]((0 until iter.next().length).toArray[Int])
  }).setName("original index").cache()


  override def data(train: Boolean): RDD[T] = {
    val _train = train
    buffer.zipPartitions(indexes)((dataIter, indexIter) => {
      val indexes = indexIter.next()
      val indexOffset = math.max(1, indexes.length)
      val localData = dataIter.next()
      val offset = if (_train) {
        RandomGenerator.RNG.uniform(0, indexOffset).toInt
      } else {
        0
      }
      new Iterator[T] {
        private val _offset = new AtomicInteger(offset)

        override def hasNext: Boolean = {
          if (_train) true else _offset.get() < localData.length
        }

        override def next(): T = {
          val i = _offset.getAndIncrement()
          if (_train) {
            // indexes is an Array, we should improve this
            // as the maximum length is limited by Int.max
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

  override def cache(): Unit = {
    buffer.count()
    indexes.count()
    isCached = true
  }

  override def unpersist(): Unit = {
    FeatureSet.logger.info(s"Unpersisting ${buffer.name}.")
    buffer.map(_.free()).count()
    buffer.unpersist()
    indexes.unpersist()
    isCached = false
  }

  override def toDistributed(): DistributedDataSet[T] = {
    new DistributedDataSetWrapper[T](this)
  }
}

object DRAMFeatureSet {
  def rdd[T: ClassTag](data: RDD[T]): DistributedFeatureSet[T] = {
    val arrayLikeRDD = data.mapPartitions(iter => {
      Iterator.single(new ArrayLikeWrapper(iter.toArray))
    }).setName(s"cached feature set: ${data.name} in DRAM" )
      .cache().asInstanceOf[RDD[ArrayLike[T]]]
    new CachedDistributedFeatureSet[T](arrayLikeRDD)
  }
}

object FeatureSet {
  val logger: Logger = LoggerFactory.getLogger(this.getClass)
  def rdd[T: ClassTag](
       data: RDD[T],
       memoryType: MemoryType = DRAM,
       dataStrategy: DataStrategy = PARTITIONED): DistributedFeatureSet[T] = {
    if (dataStrategy == PARTITIONED) {
      val nodeNumber = EngineRef.getNodeNumber()
      val repartitionedData = data.coalesce(nodeNumber, true).setName(data.name)
      memoryType match {
        case DRAM =>
          DRAMFeatureSet.rdd(repartitionedData)
        case PMEM =>
          logger.info("~~~~~~~ Caching with AEP ~~~~~~~")
          PmemFeatureSet.rdd(repartitionedData, PMEM)
        case DIRECT =>
          logger.info("~~~~~~~ Caching with DIRECT ~~~~~~~")
          PmemFeatureSet.rdd[T](repartitionedData, DIRECT)
        case _ =>
          throw new IllegalArgumentException(
            s"MemoryType: ${memoryType} is not supported at the moment")
      }
    } else {
      throw new IllegalArgumentException(
        s"DataStrategy ${dataStrategy} is not supported at the moment")
    }
  }
}
