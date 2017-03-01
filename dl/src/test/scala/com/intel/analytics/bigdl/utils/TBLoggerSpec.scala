/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.utils

import java.nio.file.Path

import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.TBLogger._
import com.intel.analytics.bigdl.visualization.tensorboard.FileWriter
import org.scalatest.{FlatSpec, Matchers}
import org.tensorflow.framework.Summary


class TBLoggerSpec extends FlatSpec with Matchers {

  "write scalar summary" should "work properly" in {
    val logdir = com.google.common.io.Files.createTempDir()
    val writer = new FileWriter(logdir.getPath)
    for (i <- 0 to 9) {
      val s = scalar("scalar", i)
      writer.addSummary(s, i + 1)
    }
    writer.flush()
    for (i <- 10 to 19) {
      val s = scalar("lr", i)
      writer.addSummary(s, i + 1)
    }
    writer.close()
  }

  "test_scalar_summary()" should "work properly" in {
    val scalar_value = 1f
    val s = scalar("test_scalar", scalar_value)
    val values = s.getValue(0)
    assert(s.getValueCount == 1)
    assert(values.getTag == "test_scalar")
    assert(values.getSimpleValue == 1.0)

    val byte_str = s.toByteArray
    val s_recovered = Summary.parseFrom(byte_str)
    assert(values.getTag == s_recovered.getValue(0).getTag())
    assert(values.getSimpleValue == s_recovered.getValue(0).getSimpleValue)
  }

  "test_log_histogram_summary" should "return write result" in {
    val logdir = com.google.common.io.Files.createTempDir()
    val writer = new FileWriter(logdir.getPath)
    for (i <- 0 until 10) {
      val mu = i * 0.1
      val sigma = 1.0
      val values = Tensor[Float](10000).apply1(_ =>
        RandomGenerator.RNG.normal(mu, sigma).toFloat
      )
      val hist = histogram("discrete_normal", values)
      writer.addSummary(hist, i + 1)
    }
    writer.flush()
    writer.close()
  }

  "saveAlexnet" should "return write result" in {
    val logdir = com.google.common.io.Files.createTempDir()
    val writer = new FileWriter(logdir.getPath)
    val alexnet = AlexNet(1000)
    val parameter = alexnet.getParameters()._1
    val hist = histogram("discrete_normal", parameter)
    writer.addSummary(hist, 1)
    writer.flush()
    writer.close()
  }
}
