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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.{SparkConf, SparkContext, SparkException}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Serial
class MetricsSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc: SparkContext = null

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "Set locale metrics" should "be able to add a new value" in {
    val metric = new Metrics
    metric.set("test", 1.0, 5)
    val result = metric.get("test")
    result._1 should be(1.0)
    result._2 should be(5)
  }

  it should "update the value if it's existed" in {
    val metric = new Metrics
    metric.set("test", 1.0, 5)
    metric.set("test", 2.0, 7)
    val result = metric.get("test")
    result._1 should be(2.0)
    result._2 should be(7)
  }

  it should "throw exception if it's a duplicated name in a distributed metric" in {
    val metric = new Metrics
    val conf = new SparkConf().setMaster("local[1]").setAppName("MetricsSpec")
    sc = new SparkContext(conf)
    metric.set("test", 10.0, sc, 5)
    intercept[IllegalArgumentException] {
      metric.set("test", 10, 5)
    }
  }

  "Set distribute metrics" should "be able to add a new value" in {
    val metric = new Metrics
    val conf = new SparkConf().setMaster("local[1]").setAppName("MetricsSpec")
    sc = new SparkContext(conf)
    metric.set("test", 1.0, sc, 5)
    val result = metric.get("test")
    result._1 should be(1.0)
    result._2 should be(5)
  }

  it should "update the value if it's existed" in {
    val metric = new Metrics
    val conf = new SparkConf().setMaster("local[1]").setAppName("MetricsSpec")
    sc = new SparkContext(conf)
    metric.set("test", 1.0, sc, 5)
    metric.set("test", 2.0, sc, 7)
    val result = metric.get("test")
    result._1 should be(2.0)
    result._2 should be(7)
  }

  it should "throw exception if it's a duplicated name in a local metric" in {
    val metric = new Metrics
    val conf = new SparkConf().setMaster("local[1]").setAppName("MetricsSpec")
    sc = new SparkContext(conf)
    metric.set("test", 10, 5)
    intercept[IllegalArgumentException] {
      metric.set("test", 10.0, sc, 5)
    }
  }

  "add value" should "work on existed a local metric" in {
    val metric = new Metrics
    metric.set("test", 1.0, 5)
    metric.add("test", 2.0)
    val result = metric.get("test")
    result._1 should be(3.0)
    result._2 should be(5)
  }

  it should "work on existed a local metric in multi-thread env" in {
    val metric = new Metrics
    metric.set("test", 1.0, 5)
    Engine.default.invokeAndWait(
      (1 to 5).map(i => () => {
        metric.add("test", i)
      })
    )
    val result = metric.get("test")
    result._1 should be(16.0)
    result._2 should be(5)
  }

  it should "throw exception when the local metric doesn't exist" in {
    val metric = new Metrics
    intercept[IllegalArgumentException] {
      metric.add("test", 10.0)
    }
  }

  it should "work on distributed metric" in {
    val metric = new Metrics
    val conf = new SparkConf().setMaster("local[5]").setAppName("MetricsSpec")
    sc = new SparkContext(conf)
    metric.set("test", 1.0, sc, 5)
    sc.parallelize((1 to 5)).map(i => metric.add("test", i)).count
    val result = metric.get("test")
    result._1 should be(16.0)
    result._2 should be(5)
  }

  it should "throw exception when the distributed metric doesn't exist" in {
    val metric = new Metrics
    val conf = new SparkConf().setMaster("local[5]").setAppName("MetricsSpec")
    sc = new SparkContext(conf)
    intercept[SparkException] {
      sc.parallelize((1 to 5)).map(i => metric.add("test", i)).count
    }
  }

  "summary" should "work" in {
    val metric = new Metrics
    val conf = new SparkConf().setMaster("local[5]").setAppName("MetricsSpec")
    sc = new SparkContext(conf)
    metric.set("test1", 1.0, sc, 5)
    metric.set("test2", 5.0, sc, 2)
    metric.summary() should be("========== Metrics Summary ==========\n" +
      "test2 : 2.5E-9 s\ntest1 : 2.0E-10 s\n=====================================")
  }

  it should "work when change the unit and scale" in {
    val metric = new Metrics
    val conf = new SparkConf().setMaster("local[5]").setAppName("MetricsSpec")
    sc = new SparkContext(conf)
    metric.set("test1", 1.0, sc, 5)
    metric.set("test2", 5.0, sc, 2)
    metric.summary("ms", 1e6) should be("========== Metrics Summary ==========\n" +
      "test2 : 2.5E-6 ms\ntest1 : 2.0000000000000002E-7 ms\n=====================================")
  }
}
