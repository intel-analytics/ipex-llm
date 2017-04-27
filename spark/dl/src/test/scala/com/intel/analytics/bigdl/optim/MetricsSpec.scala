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

import com.intel.analytics.bigdl.SparkContextSpec
import com.intel.analytics.bigdl.utils.Engine

@com.intel.analytics.bigdl.tags.Serial
class MetricsSpec extends SparkContextSpec {

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
    metric.set("test", 10.0, sc, 5)
    intercept[IllegalArgumentException] {
      metric.set("test", 10, 5)
    }
  }

  "Set distribute metrics" should "be able to add a new value" in {
    val metric = new Metrics
    metric.set("test", 1.0, sc, 5)
    val result = metric.get("test")
    result._1 should be(1.0)
    result._2 should be(5)
  }

  it should "update the value if it's existed" in {
    val metric = new Metrics
    metric.set("test", 1.0, sc, 5)
    metric.set("test", 2.0, sc, 7)
    val result = metric.get("test")
    result._1 should be(2.0)
    result._2 should be(7)
  }

  it should "throw exception if it's a duplicated name in a local metric" in {
    val metric = new Metrics
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

  it should "throw exception when the local metric isn't exsited" in {
    val metric = new Metrics
    intercept[IllegalArgumentException] {
      metric.add("test", 10.0)
    }
  }

}
