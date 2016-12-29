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

package org.apache.spark.rdd

import java.io.{IOException, ObjectOutputStream}

import org.apache.spark.util.Utils
import org.apache.spark.{Partition, SparkContext, TaskContext}

import scala.reflect.ClassTag

object ZippedPartitionsWithLocalityRDD {
  def apply[T: ClassTag, B: ClassTag, V: ClassTag]
  (rdd1: RDD[T], rdd2: RDD[B], preservesPartitioning: Boolean = false)
    (f: (Iterator[T], Iterator[B]) => Iterator[V]): RDD[V] = rdd1.withScope {
    val sc = rdd1.sparkContext
    new ZippedPartitionsWithLocalityRDD(
      sc, sc.clean(f), rdd1, rdd2, preservesPartitioning)
  }
}

class ZippedPartitionsWithLocalityRDD[A: ClassTag, B: ClassTag, V: ClassTag](
    sc: SparkContext,
    _f: (Iterator[A], Iterator[B]) => Iterator[V],
    _rdd1: RDD[A],
    _rdd2: RDD[B],
    preservesPartitioning: Boolean = false)
  extends ZippedPartitionsRDD2[A, B, V](sc, _f, _rdd1, _rdd2, preservesPartitioning) {

  override def getPartitions: Array[Partition] = {
    require(rdds.length == 2, "this is only for 2 rdd zip")
    val numParts = rdds.head.partitions.length
    if (!rdds.forall(rdd => rdd.partitions.length == numParts)) {
      throw new IllegalArgumentException("Can't zip RDDs with unequal numbers of partitions")
    }
    Array.tabulate[Partition](numParts) { i =>
      val curPrefs = rdds(0).preferredLocations(rdds(0).partitions(i))
      val prefs = rdds(1).partitions.map(rdds(1).preferredLocations(_))
      var p = 0
      var locIndex = -1
      var locs : Seq[String] = null
      while(p < prefs.length) {
        println(prefs(p))
        println(curPrefs)
        locs = prefs(p).intersect(curPrefs)
        if (!locs.isEmpty) {
          locIndex = p
          p = Integer.MAX_VALUE - 1
        }
        p += 1
      }
      require(locIndex != -1, s"can't find locality partition for partition $i")
      println(s"$i $locIndex")
      new ZippedPartitionsLocalityPartition(i, Array(i, locIndex), rdds, locs)
    }
  }

  override def compute(s: Partition, context: TaskContext): Iterator[V] = {
    tryPrepareParents()
    val partitions = s.asInstanceOf[ZippedPartitionsLocalityPartition].partitions
    f(rdd1.iterator(partitions(0), context), rdd2.iterator(partitions(1), context))
  }
}

private[spark] class ZippedPartitionsLocalityPartition(
  idx: Int,
  @transient indexes: Seq[Int],
  @transient rdds: Seq[RDD[_]],
  @transient val preferredLocations: Seq[String])
  extends Partition {

  override val index: Int = idx
  var partitionValues = rdds.zip(indexes).map{ case (rdd, i) => rdd.partitions(i) }
  def partitions: Seq[Partition] = partitionValues

  @throws(classOf[IOException])
  private def writeObject(oos: ObjectOutputStream): Unit = Utils.tryOrIOException {
    // Update the reference to parent split at the time of task serialization
    partitionValues = rdds.zip(indexes).map{ case (rdd, i) => rdd.partitions(i) }
    oos.defaultWriteObject()
  }
}


