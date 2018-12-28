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

import com.intel.analytics.bigdl.dataset.DistributedDataSet
import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.zoo.feature.common.{ArrayLike, ArrayLikeWrapper}
import com.intel.analytics.zoo.feature.pmem._
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.reflect.ClassTag


/**
 * Wrap a RDD as a DataSet.
 * @param buffer
 */
// T is the returning value type. like ByteRecord
class DistributedFeatureSet[T: ClassTag]
(buffer: RDD[ArrayLike[T]])
  extends DistributedDataSet[T] {

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
    buffer.unpersist()
    indexes.unpersist()
    isCached = false
  }
}

object DRAMFeatureSet {
  def rdd[T: ClassTag](data: RDD[T]): DistributedFeatureSet[T] = {
    val arrayLikeRDD = data.mapPartitions(iter => {
        Iterator.single(new ArrayLikeWrapper(iter.toArray))
      }).setName(s"cached feature set: ${data.name} in DRAM with PARTITIONED Strategy" )
      .cache().asInstanceOf[RDD[ArrayLike[T]]]
    new DistributedFeatureSet[T](arrayLikeRDD)
  }
}

object FeatureSet {
  val logger: Logger = LoggerFactory.getLogger(this.getClass)
  def rdd[T: ClassTag](data: RDD[T],
      memoryType: MemoryType = DRAM,
      dataStrategy: DataStrategy = PARTITIONED): DistributedFeatureSet[T] = {
    if (dataStrategy == PARTITIONED) {
      val nodeNumber = EngineRef.getNodeNumber()
      val repartitionedData = data.coalesce(nodeNumber, true)
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
