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

import java.io.StringWriter
import java.nio.charset.{Charset, StandardCharsets}
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn.{Linear, MSECriterion, Sequential}
import com.intel.analytics.bigdl.numeric.NumericDouble
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils._
import org.apache.logging.log4j.core.LoggerContext
import org.apache.logging.log4j.{Level, LogManager}
import org.apache.logging.log4j.core.appender.{ConsoleAppender, WriterAppender}
import org.apache.logging.log4j.core.config.Configurator
import org.apache.logging.log4j.core.filter.ThresholdFilter
import org.apache.logging.log4j.core.layout.PatternLayout
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.JavaConverters._

@com.intel.analytics.bigdl.tags.Serial
class LoggerFilterSpec extends FlatSpec with BeforeAndAfter with Matchers {

  var sc: SparkContext = null

  after {
    if (sc != null) { sc.stop() }
  }

  def writerAndAppender: (WriterAppender, StringWriter) = {
    // add an appender to optmClz because it's very hard to get the stdout of Log4j.
    // and `Console.setOut` or `Console.err` can't work for this case.
    // another way is to write a class extends `OutputStream`, see
    // https://sysgears.com/articles/how-to-redirect-stdout-and-stderr-writing-to-a-log4j-appender/.
    val writer = new StringWriter

    val logContext = LogManager.getContext(false).asInstanceOf[LoggerContext]
    val config = logContext.getConfiguration()
    val pattern = "%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n"
    val layout = PatternLayout.createLayout(pattern, null, config, null,
      Charset.defaultCharset, false, false, "", "")
    val filter = ThresholdFilter.createFilter(Level.ALL,
      org.apache.logging.log4j.core.Filter.Result.NEUTRAL,
      org.apache.logging.log4j.core.Filter.Result.DENY)
    val writerAppender = WriterAppender.createAppender(layout, filter, writer, "writer", true, true)
    writerAppender.start()
    config.addAppender(writerAppender)
    logContext.updateLoggers()

    // val layout = new PatternLayout("%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n")
    // val writerAppender = new WriterAppender(layout, writer)
    // writerAppender.setEncoding("UTF-8")
    // writerAppender.setThreshold(Level.ALL)
    // writerAppender.activateOptions()

    (writerAppender, writer)
  }

  ignore should "A LoggerFilter output correct info on console and bigdl.log" in {
    TestUtils.cancelOnWindows()
    val logFile = Paths.get(System.getProperty("user.dir"), "bigdl.log").toString
    val optimClz = "com.intel.analytics.bigdl.dllib.optim"

    val (writerAppender, writer) = writerAndAppender
    val logger = LogManager.getLogger(optimClz)
    if (logger.isInstanceOf[org.apache.logging.log4j.core.Logger]) {
      logger.asInstanceOf[org.apache.logging.log4j.core.Logger].addAppender(writerAppender)
    }
    val logContext = LogManager.getContext(false).asInstanceOf[LoggerContext]
    logContext.updateLoggers()
    // LogManager.getLogger(optimClz).addAppender(writerAppender)

    Files.deleteIfExists(Paths.get(logFile))
    LoggerFilter.redirectSparkInfoLogs()
    Configurator.setLevel(optimClz, Level.INFO)
    // LogManager.getLogger(optimClz).setLevel(Level.INFO)

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
      Sample[Double](featureTensor, labelTensor)
    }

    val trainSet = DataSet.rdd(data).transform(SampleToMiniBatch(recordSize/2))

    val state =
      T(
        "learningRate" -> 0.01
      )
    val criterion = MSECriterion()

    val optimizer = Optimizer(model = model,
      dataset = trainSet,
      criterion = criterion)

    optimizer.
      setState(state).
      setEndWhen(Trigger.everyEpoch).
      setOptimMethod(new SGD()).
      optimize()

    require(Files.exists(Paths.get(logFile)), s"didn't generate $logFile")

    val allString = writer.toString
    // writerAppender.close
    writerAppender.stop()

    // check the first line and the last line of BigDL
    {
      val pattern = ".*INFO.*DistriOptimizer.*caching training rdd ..."
      val firstLine = allString.split('\n')(0)
      require(firstLine.matches(pattern), s"output can't matchs the specific output\n")
    }

    {
      val pattern = s".*INFO.*DistriOptimizer.* - " + "" +
        s"\\[Epoch 1 100/100\\]\\[Iteration 2\\]\\[Wall Clock .*\\] " +
        s"Epoch finished. Wall clock time is .*ms"

      val lastLine = allString.split('\n').last
      require(lastLine.matches(pattern), s"output can't match the specific output")
    }

  }

  "A LoggerFilter generate log " should "in correct place" in {
    TestUtils.cancelOnWindows()
    val logFile = Paths.get(System.getProperty("user.dir"), "bigdl.log").toString
    val optimClz = "com.intel.analytics.bigdl.dllib.optim"

    Files.deleteIfExists(Paths.get(logFile))
    // Logger.getLogger("org").setLevel(Level.INFO)
    LoggerFilter.redirectSparkInfoLogs()
    // Logger.getLogger(optimClz).setLevel(Level.INFO)

    Configurator.setLevel("org", Level.INFO)
    Configurator.setLevel(optimClz, Level.INFO)

    sc = new SparkContext(
      Engine.init(1, 1, true).get
        .setAppName(s"LoggerFilter test")
        .set("spark.task.maxFailures", "1")
        .setMaster("local[1]")
    )

    val data = sc.parallelize(List("bigdl", "spark", "deep", "learning"))
    val y = data.map(x => (x, x.length))

    Paths.get(logFile)
    Files.exists(Paths.get(logFile)) should be (true)

    Files.deleteIfExists(Paths.get(logFile))
  }

  "A LoggerFilter generate log " should "under the place user gived" in {
    TestUtils.cancelOnWindows()
    val logFile = Paths.get(System.getProperty("java.io.tmpdir"), "bigdl.log").toString
    val optimClz = "com.intel.analytics.bigdl.dllib.optim"
    val defaultFile = Paths.get(System.getProperty("user.dir"), "bigdl.log").toString

    System.setProperty("bigdl.utils.LoggerFilter.logFile", logFile)

    Files.deleteIfExists(Paths.get(defaultFile))
    Files.deleteIfExists(Paths.get(logFile))
    // Logger.getLogger("org").setLevel(Level.INFO)
    LoggerFilter.redirectSparkInfoLogs()
    // Logger.getLogger(optimClz).setLevel(Level.INFO)

    Configurator.setLevel("org", Level.INFO)
    Configurator.setLevel(optimClz, Level.INFO)


    sc = new SparkContext(
      Engine.init(1, 1, true).get
        .setAppName(s"LoggerFilter test")
        .set("spark.task.maxFailures", "1")
        .setMaster("local[1]")
    )

    val data = sc.parallelize(List("bigdl", "spark", "deep", "learning"))
    val y = data.map(x => (x, x.length))

    Paths.get(logFile)
    Files.exists(Paths.get(logFile)) should be (true)
    Files.exists(Paths.get(defaultFile)) should be (false)

    Files.deleteIfExists(Paths.get(logFile))
    Files.deleteIfExists(Paths.get(defaultFile))
    System.clearProperty("bigdl.utils.LoggerFilter.logFile")
  }

  "A LoggerFilter generate log" should "not modify log level user defined" in {
    TestUtils.cancelOnWindows()
    val optimClz = "com.intel.analytics.bigdl.dllib.optim"
    val defaultFile = Paths.get(System.getProperty("user.dir"), "bigdl.log").toString

    Files.deleteIfExists(Paths.get(defaultFile))

    // Logger.getLogger("org").setLevel(Level.INFO)
    LoggerFilter.redirectSparkInfoLogs()
    // Logger.getLogger(optimClz).setLevel(Level.INFO)

    Configurator.setLevel("org", Level.INFO)
    Configurator.setLevel(optimClz, Level.INFO)


    val (writerAppender, writer) = writerAndAppender

    // val logger = Logger.getLogger(getClass)
    // logger.setLevel(Level.INFO)
    // logger.addAppender(writerAppender)

    val logger = LogManager.getLogger(getClass)
    if (logger.isInstanceOf[org.apache.logging.log4j.core.Logger]) {
      logger.asInstanceOf[org.apache.logging.log4j.core.Logger].addAppender(writerAppender)
    }
    val logContext = LogManager.getContext(false).asInstanceOf[LoggerContext]
    logContext.updateLoggers()

    logger.info("HELLO")



    sc = new SparkContext(
      Engine.init(1, 1, true).get
        .setAppName(s"LoggerFilter test")
        .set("spark.task.maxFailures", "1")
        .setMaster("local[1]")
    )

    val data = sc.parallelize(List("bigdl", "spark", "deep", "learning"))
    val y = data.map(x => (x, x.length))

    val allString = writer.toString
    // writerAppender.close
    writerAppender.stop()

    // check the first line and the last line of BigDL
    {
      val pattern = ".*INFO.*LoggerFilterSpec:.*HELLO"
      val firstLine = allString.split('\n')(0)
      println("-")
      require(firstLine.matches(pattern), s"output can't matchs the specific output")
    }

    Files.deleteIfExists(Paths.get(defaultFile))
  }

  "A LoggerFilter generate log" should "be controled by the property" in {
    val optimClz = "com.intel.analytics.bigdl.dllib.optim"
    val defaultFile = Paths.get(System.getProperty("user.dir"), "bigdl.log").toString

    System.setProperty("bigdl.utils.LoggerFilter.disable", "true")

    Files.deleteIfExists(Paths.get(defaultFile))

    // Logger.getLogger("org").setLevel(Level.INFO)
    LoggerFilter.redirectSparkInfoLogs()
    // Logger.getLogger(optimClz).setLevel(Level.INFO)

    Configurator.setLevel("org", Level.INFO)
    Configurator.setLevel(optimClz, Level.INFO)

    sc = new SparkContext(
      Engine.init(1, 1, true).get
        .setAppName(s"LoggerFilter test")
        .set("spark.task.maxFailures", "1")
        .setMaster("local[1]")
    )

    val data = sc.parallelize(List("bigdl", "spark", "deep", "learning"))
    val y = data.map(x => (x, x.length))

    Files.exists(Paths.get(defaultFile)) should be (false)
    System.clearProperty("bigdl.utils.LoggerFilter.disable")
  }

  "A LoggerFilter user's log" should "be in log file" in {
    val defaultFile = Paths.get(System.getProperty("user.dir"), "bigdl.log").toString
    LoggerFilter.redirectSparkInfoLogs()

    val info = "bigdl info message"
    val warn = "bigdl warn message"
    val error = "bigdl error message"

    LogManager.getLogger(getClass).info(info)
    LogManager.getLogger(getClass).warn(warn)
    LogManager.getLogger(getClass).error(error)

    val lines = Files.readAllLines(Paths.get(defaultFile), StandardCharsets.UTF_8)

    // lines.size() should be (3)
    // lines.get(0).contains(info) should be (true)
    // lines.get(1).contains(warn) should be (true)
    // lines.get(2).contains(error) should be (true)

    Files.deleteIfExists(Paths.get(defaultFile))
    Files.exists(Paths.get(defaultFile)) should be (false)
  }

  "A LoggerFilter disable spark log" should "not generate spark logs in file" in {
    TestUtils.cancelOnWindows()
    val defaultFile = Paths.get(System.getProperty("user.dir"), "bigdl.log").toString
    System.setProperty("bigdl.utils.LoggerFilter.enableSparkLog", "false")
    LoggerFilter.redirectSparkInfoLogs()

    sc = new SparkContext(
      Engine.init(1, 1, true).get
        .setAppName(s"LoggerFilter test")
        .set("spark.task.maxFailures", "1")
        .setMaster("local[1]")
    )

    val data = sc.parallelize(List("bigdl", "spark", "deep", "learning"))
    val y = data.map(x => (x, x.length)).count()

    val lines = Files.readAllLines(Paths.get(defaultFile), StandardCharsets.UTF_8).asScala
    lines.exists(_.contains("DAGScheduler")) should be (false)

    Files.deleteIfExists(Paths.get(defaultFile))
    Files.exists(Paths.get(defaultFile)) should be (false)
    System.clearProperty("bigdl.utils.LoggerFilter.enableSparkLog")
  }
}
