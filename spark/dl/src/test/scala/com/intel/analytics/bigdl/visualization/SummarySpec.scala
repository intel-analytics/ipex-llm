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

package com.intel.analytics.bigdl.visualization

import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, TestUtils}
import Summary._
import com.intel.analytics.bigdl.visualization.tensorboard.{FileReader, FileWriter}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import org.tensorflow.framework

@com.intel.analytics.bigdl.tags.Serial
class SummarySpec extends FlatSpec with Matchers with BeforeAndAfter {

  before {
    System.setProperty("bigdl.localMode", "false")
    Engine.init(1, 4, true)
  }

  after {
    System.clearProperty("bigdl.localMode")
  }

  "write scalar summary" should "work properly" in {
    val logdir = com.google.common.io.Files.createTempDir()
    val writer = new FileWriter(logdir.getPath)
    for (i <- 0 to 9) {
      val s = scalar("scalar", i)
      writer.addSummary(s, i + 1)
    }
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
    val s_recovered = framework.Summary.parseFrom(byte_str)
    assert(values.getTag == s_recovered.getValue(0).getTag())
    assert(values.getSimpleValue == s_recovered.getValue(0).getSimpleValue)
  }

  "test_log_histogram_summary" should "return write result" in {
    RandomGenerator.RNG.setSeed(1)
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
    writer.close()
  }

  "saveAlexnet" should "return write result" in {
    val logdir = com.google.common.io.Files.createTempDir()
    val writer = new FileWriter(logdir.getPath)
    val alexnet = AlexNet(1000)
    val parameter = alexnet.getParameters()._1
    val hist = histogram("discrete_normal", parameter)
    writer.addSummary(hist, 1)
    writer.close()
  }

  "read/write event file" should "work properly" in {
    TestUtils.cancelOnWindows()
    val logdir = com.google.common.io.Files.createTempDir()
    val writer = new FileWriter(logdir.getPath, 100)
    for (i <- 0 to 9) {
      val s = scalar("scalar", i)
      writer.addSummary(s, i + 1)
    }
    for (i <- 10 to 19) {
      val s = scalar("lr", i)
      writer.addSummary(s, i + 1)
    }
    for (i <- 0 to 9) {
      val s = scalar("lr", i)
      writer.addSummary(s, i + 1)
    }
    writer.close()
    Thread.sleep(1000) // Waiting for writer.
    val tbReader = FileReader.list(logdir.getPath)
    val result = FileReader.readScalar(tbReader(0), "lr")
    result.length should be (20)
    for (i <- 0 to 19) {
      result(i)._1 should be (i + 1)
      result(i)._2 should be (i)
    }
  }

  "read event file with a non-existent tag" should "return a empty array" in {
    TestUtils.cancelOnWindows()
    val logdir = com.google.common.io.Files.createTempDir()
    val writer = new FileWriter(logdir.getPath, 100)
    for (i <- 0 to 9) {
      val s = scalar("scalar", i)
      writer.addSummary(s, i + 1)
    }
    writer.close()
    Thread.sleep(1000) // Waiting for writer.
    val tbReader = FileReader.list(logdir.getPath)
    val result = FileReader.readScalar(tbReader(0), "lr")
    result.length should be(0)
  }

  "FileReader.list" should "work properly" in {
    TestUtils.cancelOnWindows()
    val logdir = com.google.common.io.Files.createTempDir()
    val writer1 = new FileWriter(logdir.getPath, 100)
    for (i <- 0 to 9) {
      val s = scalar("scalar1", i)
      writer1.addSummary(s, i + 1)
    }
    writer1.close()
    Thread.sleep(1000) // Waiting for writer.
    val writer2 = new FileWriter(logdir.getPath, 100)
    for (i <- 0 to 19) {
      val s = scalar("scalar2", i)
      writer2.addSummary(s, i + 1)
    }
    writer2.close()
    Thread.sleep(1000) // Waiting for writer.
    val tbFiles = FileReader.listFiles(logdir.getPath)
    tbFiles.length should be (2)
    val tbFolder = FileReader.list(logdir.getPath)
    tbFolder.length should be (1)
    tbFolder(0).replace("file:", "") should be (logdir.getPath)
  }

  "FileReader read from five Files" should "work properly" in {
    TestUtils.cancelOnWindows()
    val numFile = 5
    val logdir = com.google.common.io.Files.createTempDir()
    for (i <- 1 to numFile) {
      val writer = new FileWriter(logdir.getPath, 10)
      for (j <- 0 to i) {
        val s = scalar(s"scalar$i", j)
        writer.addSummary(s, j + 1)
      }
      writer.close()
      Thread.sleep(1000) // sleep to get a different filename
    }
    Thread.sleep(1000) // Wait for the writing
    val tbFiles = FileReader.listFiles(logdir.getPath)
    tbFiles.length should be (numFile)
    val tbFolder = FileReader.list(logdir.getPath)
    tbFolder.length should be (1)
    tbFolder(0).replace("file:", "") should be (logdir.getPath)
    for (i <- 1 to numFile) {
      val result = FileReader.readScalar(tbFolder(0), s"scalar$i")
      result.length should be (i + 1)
      for (j <- 0 to i) {
        result(j)._2 should be (j)
        result(j)._1 should be (j + 1)
      }
    }

  }
}
