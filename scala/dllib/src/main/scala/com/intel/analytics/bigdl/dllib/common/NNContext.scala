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

package com.intel.analytics.zoo.common

import java.io.InputStream
import java.util.Properties

import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.Logger
import org.apache.spark.{SPARK_VERSION, SparkConf, SparkContext, SparkException}
import sys.env

/**
 * [[NNContext]] wraps a spark context in Analytics Zoo.
 *
 */
object NNContext {

  private val logger = Logger.getLogger(getClass)

  private[zoo] def checkSparkVersion(reportWarning: Boolean = false) = {
    checkVersion(SPARK_VERSION, ZooBuildInfo.spark_version, "Spark", reportWarning)
  }

  private[zoo] def checkScalaVersion(reportWarning: Boolean = false) = {
    checkVersion(scala.util.Properties.versionNumberString,
      ZooBuildInfo.scala_version, "Scala", reportWarning, level = 2)
  }

  private def checkVersion(
                            runtimeVersion: String,
                            compileTimeVersion: String,
                            project: String,
                            reportWarning: Boolean = false,
                            level: Int = 1): Unit = {
    val Array(runtimeMajor, runtimeFeature, runtimeMaintenance) =
      runtimeVersion.split("\\.").map(_.toInt)
    val Array(compileMajor, compileFeature, compileMaintenance) =
      compileTimeVersion.split("\\.").map(_.toInt)

    if (runtimeVersion != compileTimeVersion) {
      val warnMessage = s"The compile time $project version is not compatible with" +
        s" the runtime $project version. Compile time version is $compileTimeVersion," +
        s" runtime version is $runtimeVersion. "
      val errorMessage = s"\nIf you want to bypass this check, please set" +
        s"spark.analytics.zoo.versionCheck to false, and if you want to only" +
        s"report a warning message, please set spark.analytics.zoo" +
        s".versionCheck.warning to true."
      val diffLevel = if (runtimeMajor != compileMajor) {
        1
      } else if (runtimeFeature != compileFeature) {
        2
      } else {
        3
      }
      if (diffLevel <= level && !reportWarning) {
        Utils.logUsageErrorAndThrowException(warnMessage + errorMessage)
      }
      logger.warn(warnMessage)
    } else {
      logger.info(s"$project version check pass")
    }
  }

  private[zoo] object ZooBuildInfo {

    val (
      analytics_zoo_verion: String,
      spark_version: String,
      scala_version: String,
      java_version: String) = {

      val resourceStream = Thread.currentThread().getContextClassLoader.
        getResourceAsStream("zoo-version-info.properties")

      try {
        val unknownProp = "<unknown>"
        val props = new Properties()
        props.load(resourceStream)
        (
          props.getProperty("analytics_zoo_verion", unknownProp),
          props.getProperty("spark_version", unknownProp),
          props.getProperty("scala_version", unknownProp),
          props.getProperty("java_version", unknownProp)
        )
      } catch {
        case npe: NullPointerException =>
          throw new RuntimeException("Error while locating file zoo-version-info.properties, " +
            "if you are using an IDE to run your program, please make sure the mvn" +
            " generate-resources phase is executed and a zoo-version-info.properties file" +
            " is located in zoo/target/extra-resources", npe)
        case e: Exception =>
          throw new RuntimeException("Error loading properties from zoo-version-info.properties", e)
      } finally {
        if (resourceStream != null) {
          try {
            resourceStream.close()
          } catch {
            case e: Exception =>
              throw new SparkException("Error closing zoo build info resource stream", e)
          }
        }
      }
    }
  }

  /**
   * Creates or gets a SparkContext with optimized configuration for BigDL performance.
   * The method will also initialize the BigDL engine.
   *
   * Note: if you use spark-shell or Jupyter notebook, as the Spark context is created
   * before your code, you have to set Spark conf values through command line options
   * or properties file, and init BigDL engine manually.
   *
   * @param conf User defined Spark conf
   * @param appName name of the current context
   * @return Spark Context
   */
  def initNNContext(conf: SparkConf, appName: String): SparkContext = {
    val zooConf = createSparkConf(conf)
    initConf(zooConf)

    if (appName != null) {
      zooConf.setAppName(appName)
    }
    if (zooConf.getBoolean("spark.analytics.zoo.versionCheck", defaultValue = false)) {
      val reportWarning =
        zooConf.getBoolean("spark.analytics.zoo.versionCheck.warning", defaultValue = false)
      checkSparkVersion(reportWarning)
      checkScalaVersion(reportWarning)
    }
    val sc = SparkContext.getOrCreate(zooConf)
    Engine.init
    sc
  }

  /**
   * Creates or gets SparkContext with optimized configuration for BigDL performance.
   * The method will also initialize the BigDL engine.
   *
   * Note: if you use spark-shell or Jupyter notebook, as the Spark context is created
   * before your code, you have to set Spark conf values through command line options
   * or properties file, and init BigDL engine manually.
   *
   * @param conf User defined Spark conf
   * @return Spark Context
   */
  def initNNContext(conf: SparkConf): SparkContext = {
    initNNContext(conf = conf, appName = null)
  }

  /**
   * Creates or gets a SparkContext with optimized configuration for BigDL performance.
   * The method will also initialize the BigDL engine.
   *
   * Note: if you use spark-shell or Jupyter notebook, as the Spark context is created
   * before your code, you have to set Spark conf values through command line options
   * or properties file, and init BigDL engine manually.
   *
   * @param appName name of the current context
   * @return Spark Context
   */
  def initNNContext(appName: String): SparkContext = {
    initNNContext(conf = null, appName = appName)
  }

  def initNNContext(): SparkContext = {
    initNNContext(null, null)
  }

  /**
   * Read spark conf values from spark-analytics-zoo.conf
   */
  private[zoo] def readConf: Seq[(String, String)] = {
    val stream: InputStream = getClass.getResourceAsStream("/spark-analytics-zoo.conf")
    val lines = scala.io.Source.fromInputStream(stream)
      .getLines.filter(_.startsWith("spark")).toArray

    // For spark 1.5, we observe nio block manager has better performance than netty block manager
    // So we will force set block manager to nio. If user don't want this, he/she can set
    // bigdl.network.nio == false to customize it. This configuration/blcok manager setting won't
    // take affect on newer spark version as the nio block manger has been removed
    lines.map(_.split("\\s+")).map(d => (d(0), d(1))).toSeq
      .filter(_._1 != "spark.shuffle.blockTransferService" ||
        System.getProperty("bigdl.network.nio", "true").toBoolean)
  }

  /**
   * Spark conf with pre-set env
   * Currently, focus on KMP_AFFINITY, KMP_BLOCKTIME
   * KMP_SETTINGS, OMP_NUM_THREADS and ZOO_NUM_MKLTHREADS
   *
   * @param zooConf SparkConf
   */
  private[zoo] def initConf(zooConf: SparkConf) : Unit = {
    // check env and set spark conf
    // Set default value
    var kmpAffinity = "granularity=fine,compact,1,0"
    var kmpBlockTime = "0"
    var kmpSettings = "1"
    var ompNumThreads = "1"
    // Set value with env
    if (env.contains("KMP_AFFINITY")) {
      kmpAffinity = env("KMP_AFFINITY")
    }
    if (env.contains("KMP_BLOCKTIME")) {
      kmpBlockTime = env("KMP_BLOCKTIME")
    }
    if (env.contains("KMP_SETTINGS")) {
      kmpSettings = env("KMP_SETTINGS")
    }
    if (env.contains("OMP_NUM_THREADS")) {
      ompNumThreads = env("OMP_NUM_THREADS")
    } else if (env.contains("ZOO_NUM_MKLTHREADS")) {
      if (env("ZOO_NUM_MKLTHREADS").equalsIgnoreCase("all")) {
        val cores = Runtime.getRuntime.availableProcessors()
        ompNumThreads = cores.toString
      } else {
        ompNumThreads = env("ZOO_NUM_MKLTHREADS")
      }
    }
    // Set Spark Conf
    zooConf.setExecutorEnv("KMP_AFFINITY", kmpAffinity)
    zooConf.setExecutorEnv("KMP_BLOCKTIME", kmpBlockTime)
    zooConf.setExecutorEnv("KMP_SETTINGS", kmpSettings)
    zooConf.setExecutorEnv("OMP_NUM_THREADS", ompNumThreads)

  }

  def createSparkConf(existingConf: SparkConf = null) : SparkConf = {
    var _conf = existingConf
    if (_conf == null) {
      _conf = new SparkConf()
    }
    readConf.foreach(c => _conf.set(c._1, c._2))
    _conf
  }
}

