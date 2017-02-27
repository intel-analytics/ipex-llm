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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericDouble
import com.intel.analytics.bigdl.optim.{Optimizer, SGD, Trigger}
import com.intel.analytics.bigdl.nn.{Linear, MSECriterion, Sequential}
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample, SampleToBatch}

import scala.io.Source
import java.io.StringWriter
import java.nio.file.{Files, Paths}

import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger, PatternLayout, WriterAppender}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Serial
class LoggerFilterSpec extends FlatSpec with BeforeAndAfter with Matchers {

  var sc: SparkContext = null

  after {
    if (sc != null) { sc.stop() }
  }

  "A LoggerFilter" should "output correct info on console and /tmp/bigdl.log" in {
    val logFile = "/tmp/bigdl.log"
    val optimClz = "com.intel.analytics.bigdl.optim"

    // add an appender to optmClz because it's very hard to get the stdout of Log4j.
    // and `Console.setOut` or `Console.err` can't work for this case.
    // another way is to write a class extends `OutputStream`, see
    // https://sysgears.com/articles/how-to-redirect-stdout-and-stderr-writing-to-a-log4j-appender/.
    val writer = new StringWriter
    val layout = new PatternLayout("%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n")
    val writerAppender = new WriterAppender(layout, writer)
    writerAppender.setEncoding("UTF-8")
    writerAppender.setThreshold(Level.ALL)
    writerAppender.activateOptions()
    Logger.getLogger(optimClz).addAppender(writerAppender)

    Files.deleteIfExists(Paths.get(logFile))
    Logger.getLogger("org").setLevel(Level.INFO)
    LoggerFilter.redirectSparkInfoLogs()
    Logger.getLogger(optimClz).setLevel(Level.INFO)

    val layer = Linear(10, 1)
    val model = Sequential()
      .add(layer)
    model.reset()

    sc = new SparkContext(
      Engine.init(1, 1, true).get
        .setAppName(s"LoggerFilter test")
        .set("spark.task.maxFailures", "1")
        .setMaster("local[1]")
    )

    val maxEpoch = 1
    val recordSize = 100
    val inputDim = 10

    val data = sc.range(0, recordSize, 1, 1).map { _ =>
      val featureTensor = Tensor(inputDim).rand()
      val labelTensor = Tensor(1).rand()
      new Sample[Double](featureTensor, labelTensor)
    }

    val trainSet = DataSet.rdd(data).transform(SampleToBatch(recordSize/2))

    val state =
      T(
        "learningRate" -> 0.01
      )
    val criterion = MSECriterion()

    val optimizer = Optimizer[Double, MiniBatch[Double]](model, trainSet, criterion)

    optimizer.
      setState(state).
      setEndWhen(Trigger.everyEpoch).
      setOptimMethod(new SGD()).
      optimize()

    require(Files.exists(Paths.get(logFile)), s"doesn't generate $logFile")

    // only check the first line of the log
    {
      val pattern = ".*INFO.*SparkContext:.*Running Spark version.*"
      val firstLine = Source.fromFile(logFile).getLines.next()
      require(firstLine.matches(pattern), s"$logFile can't matchs the specific output")
      Files.deleteIfExists(Paths.get(logFile))
    }

    val allString = writer.toString
    writerAppender.close

    // check the first line and the last line of BigDL
    {
      val pattern = ".*INFO.*DistriOptimizer.*Cache thread models..."
      val firstLine = allString.split('\n')(0)
      require(firstLine.matches(pattern), s"output can't matchs the specific output")
    }

    {
      val pattern = s".*INFO.*DistriOptimizer.* - " + "" +
        s"\\[Epoch 1 50/100\\]\\[Iteration 2\\]\\[Wall Clock .*\\] " +
        s"Epoch finished. Wall clock time is .*ms"

      val lastLine = allString.split('\n').last
      require(lastLine.matches(pattern), s"output can't matchs the specific output")
    }

  }
}
