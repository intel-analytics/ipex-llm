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

import com.intel.analytics.zoo.common.NNContext
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


class FeatureSetSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc : SparkContext = _

  before {
    val conf = new SparkConf().setAppName("Test Feature Set").setMaster("local[1]")
    sc = NNContext.initNNContext(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "FeatureSet" should "iterate in sequential order without shuffle" in {

    val rdd = sc.parallelize(0 until 10, numSlices = 1)

    val fs = FeatureSet.rdd(rdd, sequentialOrder = true, shuffle = false)

    val data = fs.data(train = true)
    val seq = for (i <- 0 until 10) yield {
      data.first()
    }

    assert(seq == (0 until 10))
    fs.unpersist()
  }


  "FeatureSet" should "iterate in sequential order with shuffle" in {
    val rdd = sc.parallelize(0 until 10, numSlices = 1)

    val fs = FeatureSet.rdd(rdd, sequentialOrder = true)
    fs.shuffle()
    val data = fs.data(train = true)
    val set = scala.collection.mutable.Set[Int]()
    set ++= (0 until 10)

    val firstRound = for (i <- 0 until 10) yield {
      val value = data.first()

      set -= value
    }

    assert(firstRound != (0 until 10))
    assert(set.isEmpty)
    fs.unpersist()
  }
}
