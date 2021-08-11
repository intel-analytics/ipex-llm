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

package com.intel.analytics.bigdl.utils

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.ZippedPartitionsWithLocalityRDD
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Serial
class ZippedPartitionsWithLocalityRDDSpec extends SparkContextLifeCycle with Matchers {
  override def coreNumber: Int = 4
  override def appName: String = "ZippedPartitionsWithLocalityRDDSpec"

  "two uncached rdd zip partition" should "not throw exception" in {
    val rdd1 = sc.parallelize((1 to 100), 4)
    val rdd2 = sc.parallelize((1 to 100), 4)
      ZippedPartitionsWithLocalityRDD(rdd1, rdd2)((iter1, iter2) => {
        iter1.zip(iter2)
      }).count()
  }

  "one uncached rdd zip partition" should "not throw exception" in {
    val rdd1 = sc.parallelize((1 to 100), 4).cache()
    val rdd2 = sc.parallelize((1 to 100), 4)
      ZippedPartitionsWithLocalityRDD(rdd1, rdd2)((iter1, iter2) => {
        iter1.zip(iter2)
      }).count()
  }

  "two cached rdd zip partition" should "should be zip" in {
    val rdd1 = sc.parallelize((1 to 100), 4).repartition(4).cache()
    val rdd2 = sc.parallelize((1 to 100), 4).repartition(4).cache()

    rdd1.count()
    rdd2.count()
    rdd2.count() // need to count twice

    ZippedPartitionsWithLocalityRDD(rdd1, rdd2)((iter1, iter2) => {
      iter1.zip(iter2)
    }).count()
  }
}
