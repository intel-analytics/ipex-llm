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

import com.intel.analytics.bigdl.dataset.{ByteRecord, Sample}
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.{CachedDistributedFeatureSet, DistributedFeatureSet}
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
  override def free(): Unit = {
    records.free()
  }
}

private[zoo] class SampleConverter(
    memoryType: MemoryType = PMEM) extends NativeArrayConverter[Sample[Float]] {

  override def getBytesPerRecord(sample: Sample[Float]): Long = {
    sample.getData().length * 4
  }

  override def toArray(
      recordIterator: Iterator[Sample[Float]],
      countPerPartition: Iterator[(Int, Long)]): Iterator[ArrayLike[Sample[Float]]] = {
    val count = countPerPartition.next()
    val nativeArray = new VarLenFloatsArray(count._1, count._2,
          memoryType = memoryType)

    val featureSizes = new Array[Array[Array[Int]]](count._1)
    val labelSizes = new Array[Array[Array[Int]]](count._1)
    var i = 0
    while(recordIterator.hasNext) {
      val data = recordIterator.next()
      nativeArray.set(i, data.getData())
      featureSizes(i) = data.getFeatureSize()
      labelSizes(i) = data.getLabelSize()
      i += 1
    }
    Iterator.single(SampleArray(nativeArray, featureSizes, labelSizes))
  }
}

private[zoo] case class SampleArray(
    samples: VarLenFloatsArray,
    featureSizes: Array[Array[Array[Int]]],
    labelSizes: Array[Array[Array[Int]]]) extends ArrayLike[Sample[Float]] {
  override def length: Int = {
    samples.recordNum
  }

  override def apply(i: Int): Sample[Float] = {
    Sample[Float](samples.get(i), featureSizes(i), labelSizes(i))
  }

  override def free(): Unit = {
    samples.free()
  }
}

private[zoo] class ImageFeatureConverter(
      memoryType: MemoryType = PMEM) extends NativeArrayConverter[ImageFeature] {

  override def getBytesPerRecord(imageFeature: ImageFeature): Long = {
    imageFeature.bytes().length
  }

  override def toArray(
        recordIterator: Iterator[ImageFeature],
        countPerPartition: Iterator[(Int, Long)]): Iterator[ArrayLike[ImageFeature]] = {
    val count = countPerPartition.next()
    val nativeArray = new VarLenBytesArray(count._1, count._2,
      memoryType = memoryType)
    // cache ImageFeature without bytes.
    val metrics = new Array[ImageFeature](count._1)
    var i = 0
    while(recordIterator.hasNext) {
      // Move bytes in ImageFeature to PMEM, then remove bytes in ImageFeature to minimize
      // memory used in DRAM.
      val data = recordIterator.next()
      require(data.contains(ImageFeature.bytes),
        s"Only support cache ImageFeature's bytes" +
        s"to PMEM, but no bytes data found, please check your data.")
      nativeArray.set(i, data.bytes())
      data.update(ImageFeature.bytes, null)
      metrics(i) = data
      i += 1
    }
    Iterator.single(ImageFeatureArray(nativeArray, metrics))
  }
}

/**
 * Cached ImageFeatures in PMEM.
 * @param bytesData bytes in PMEM.
 * @param metrics ImageFeature without bytes, just some metrics.
 */
private[zoo] case class ImageFeatureArray(
      bytesData: VarLenBytesArray,
      metrics: Array[ImageFeature]) extends ArrayLike[ImageFeature] {
  override def length: Int = {
    bytesData.recordNum
  }
  override def apply(i: Int): ImageFeature = {
    val data = metrics(i).clone()
    data.update("bytes", bytesData.get(i))
    data
  }
  override def free(): Unit = {
    bytesData.free()
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
      // Add a hooker to offset the pmem resource
      Runtime.getRuntime().addShutdownHook(new Thread() {
        override def run(): Unit = NativeArray.free()
      })
      nativeArrayConverter.toArray(dataIter, countIter)
    }.setName(s"FeatureSet: ${data.name} cached in PMEM")
      .cache()
    new CachedDistributedFeatureSet[T](arrayRDD.asInstanceOf[RDD[ArrayLike[T]]])
  }

  def rdd[T: ClassTag](data: RDD[T],
      memoryType: MemoryType = PMEM): DistributedFeatureSet[T] = {
    var clazz: ClassTag[T] = implicitly[ClassTag[T]]
    implicitly[ClassTag[T]].runtimeClass match {
      case t if t == classOf[ByteRecord] =>
        rdd[ByteRecord](data.asInstanceOf[RDD[ByteRecord]],
          new ByteRecordConverter(memoryType)).asInstanceOf[DistributedFeatureSet[T]]
      case t if t == classOf[Sample[Float]] =>
        rdd[Sample[Float]](data.asInstanceOf[RDD[Sample[Float]]],
          new SampleConverter(memoryType)).asInstanceOf[DistributedFeatureSet[T]]
      case t if t == classOf[ImageFeature] =>
        rdd[ImageFeature](data.asInstanceOf[RDD[ImageFeature]],
          new ImageFeatureConverter(memoryType)).asInstanceOf[DistributedFeatureSet[T]]
      case _ =>
        throw new IllegalArgumentException(
          s"${implicitly[ClassTag[T]].runtimeClass} is not supported for now")
    }
  }
}
