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

package com.intel.analytics.zoo.feature.pmem

import com.intel.analytics.bigdl.dataset.ByteRecord
import com.intel.analytics.zoo.feature.DistributedFeatureSet
import com.intel.analytics.zoo.feature.common.ArrayLike
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

private[zoo] abstract class NativeArrayConverter[T: ClassTag]
  extends Serializable {

  def getBytesPerRecord(record: T): Long

  def toArray(recordIterator: Iterator[T],
      countPerPartition: Iterator[(Int, Long)]): Iterator[ArrayLike[T]]
}

private[zoo] class ByteRecordConverter(
    memoryType: MemoryType = PMEM) extends NativeArrayConverter[ByteRecord] {

  override def getBytesPerRecord(byteRecord: ByteRecord): Long = {
    byteRecord.data.length
  }

  override def toArray(recordIterator: Iterator[ByteRecord],
      countPerPartition: Iterator[(Int, Long)]): Iterator[ArrayLike[ByteRecord]] = {
      val count = countPerPartition.next()
      val nativeArray = new VarLenBytesArray(count._1, count._2,
        memoryType = memoryType)
      val labels = new Array[Float](count._1)
      var i = 0
      while(recordIterator.hasNext) {
        val data = recordIterator.next()
        nativeArray.set(i, data.data)
        labels(i) = data.label
        i += 1
      }
      Iterator.single(ByteRecordArray(nativeArray, labels))
    }
}

private[zoo] case class ByteRecordArray(records: VarLenBytesArray,
    label: Array[Float]) extends ArrayLike[ByteRecord] {
  override def length: Int = {
    records.recordNum
  }
  override def apply(i: Int): ByteRecord = {
    ByteRecord(records.get(i), label(i.toInt))
  }
}

object PmemFeatureSet {

  private def rdd[T: ClassTag](data: RDD[T],
      nativeArrayConverter: NativeArrayConverter[T]):
  DistributedFeatureSet[T] = {
    val countPerPartition = data.mapPartitions { iter =>
      require(iter.hasNext)
      var totalBytes: Long = 0L
      var totalRecordNum = 0
      while (iter.hasNext) {
        val record = iter.next()
        totalRecordNum += 1
        totalBytes += nativeArrayConverter.getBytesPerRecord(record)
      }
      Iterator.single((totalRecordNum, totalBytes))
    }
    val arrayRDD = data.zipPartitions(countPerPartition) { (dataIter, countIter) =>
      nativeArrayConverter.toArray(dataIter, countIter)
    }.setName(s"FeatureSet: ${data.name} cached in PMEM")
      .cache()
    new DistributedFeatureSet[T](arrayRDD.asInstanceOf[RDD[ArrayLike[T]]])
  }

  def rdd[T: ClassTag](data: RDD[T],
      memoryType: MemoryType = PMEM): DistributedFeatureSet[T] = {
    var clazz: ClassTag[T] = implicitly[ClassTag[T]]
    implicitly[ClassTag[T]].runtimeClass match {
      case t if t == classOf[ByteRecord] =>
        rdd[ByteRecord](data.asInstanceOf[RDD[ByteRecord]],
          new ByteRecordConverter(memoryType)).asInstanceOf[DistributedFeatureSet[T]]
      case _ =>
        throw new IllegalArgumentException(
          s"${implicitly[ClassTag[T]].runtimeClass} is not supported for now")
    }
  }
}
