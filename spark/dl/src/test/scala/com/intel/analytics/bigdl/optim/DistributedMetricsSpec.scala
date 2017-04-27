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
import org.apache.spark.SparkException

@com.intel.analytics.bigdl.tags.Serial
class DistributedMetricsSpec extends SparkContextSpec {

  override def getCoreNumber: Int = 5
 
  it should "work on distributed metric" in {
    val metric = new Metrics
    metric.set("test", 1.0, sc, 5)
    sc.parallelize((1 to 5)).map(i => metric.add("test", i)).count
    val result = metric.get("test")
    result._1 should be(16.0)
    result._2 should be(5)
  }

  it should "throw exception when the distributed metric isn't exsited" in {
    val metric = new Metrics
    intercept[SparkException] {
      sc.parallelize((1 to 5)).map(i => metric.add("test", i)).count
    }
  }

  "summary" should "work" in {
    val metric = new Metrics
    metric.set("test1", 1.0, sc, 5)
    metric.set("test2", 5.0, sc, 2)
    metric.summary() should be("========== Metrics Summary ==========\n" +
      "test2 : 2.5E-9 s\ntest1 : 2.0E-10 s\n=====================================")
  }

  it should "work when change the unit and scale" in {
    val metric = new Metrics
    metric.set("test1", 1.0, sc, 5)
    metric.set("test2", 5.0, sc, 2)
    metric.summary("ms", 1e6) should be("========== Metrics Summary ==========\n" +
      "test2 : 2.5E-6 ms\ntest1 : 2.0000000000000002E-7 ms\n=====================================")
  }
}
