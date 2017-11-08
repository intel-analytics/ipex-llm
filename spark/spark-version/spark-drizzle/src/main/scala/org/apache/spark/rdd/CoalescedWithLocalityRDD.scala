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

package org.apache.spark.rdd

import java.io.{IOException, ObjectOutputStream}

import org.apache.spark.util.Utils
import org.apache.spark.{Partition, SparkContext}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object CoalescedWithLocalityRDD {
  def apply[T: ClassTag](rdd: RDD[T], partitionNum: Int): RDD[T] = rdd.withScope {
    new CoalescedWithLocalityRDD(rdd, partitionNum)
  }
}

class CoalescedWithLocalityRDD[T: ClassTag](
  _rdd: RDD[T],
  partitionNum: Int)
  extends CoalescedRDD[T](_rdd, partitionNum) {
  override def getPartitions: Array[Partition] = {
    val pc = new DefaultPartitionCoalescer(1.0)

    pc.coalesce(partitionNum, _rdd).zipWithIndex.map {
      case (pg, i) =>
        val ids = pg.partitions.map(_.index).toArray
        new CoalescedRDDPartition(i, _rdd, ids, pg.prefLoc)
    }
  }
}