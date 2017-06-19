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

import java.io.InputStream
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext}

import com.intel.analytics.bigdl.mkl.MKL

/**
 * define engine type trait
 */
sealed trait EngineType

case object MklBlas extends EngineType


object Engine {
  @deprecated(
    "See https://github.com/intel-analytics/BigDL/wiki/Programming-Guide#engine",
    "0.1.0")
  def init(nExecutor: Int,
           executorCores: Int,
           onSpark: Boolean): Option[SparkConf] = {
    logger.warn("Engine.init(nExecutor, executorCores, onSpark) is deprecated. " +
      "Please refer to " +
      "https://github.com/intel-analytics/BigDL/wiki/Programming-Guide#engine")
    setNodeAndCore(nExecutor, executorCores)
    val res = if (onSpark) {
      require(localMode == false,
        s"Engine.init: bigdl.localMode should not be set while onSpark is " +
          s"true. Please set correct java property.")
      Some(createSparkConf())
    } else {
      require(localMode == true,
        s"Engine.init: bigdl.localMode should be set while onSpark is " +
          s"false. Please set correct java property.")
      None
    }
    res
  }

  /**
   * BigDL need some Spark conf values to be set correctly to have better performance.
   *
   * This method will create a SparkConf, or use the existing one if you provide one.
   * Populate it with correct values.
   *
   * We recommend you use this method instead of setting spark conf values directly. This can
   * make the Spark conf values changes transparent to you. However, if you use spark-shell or
   * Jupyter notebook, as the Spark context is created before your code, you have to
   * set them directly (through command line options or properties file)
   *
   * @return
   */
  def createSparkConf(exisitingConf : SparkConf = null) : SparkConf = {
    var _conf = exisitingConf
    if (_conf == null) {
      _conf = new SparkConf()
    }
    readConf.foreach(c => _conf.set(c._1, c._2))
    _conf
  }

  /**
   * This method should be call before any BigDL procedure and after the Spark context is created.
   *
   * BigDL needs some Spark conf values to be set correctly to have a better performance. There's
   * also multi-thread engines so executor number and core number per executor need to be known
   * to set the parameter of these engines correctly.
   *
   * The method can set parameters of multi-thread engines, verify spark conf values of an
   * existing spark context.
   */
  def init: Unit = this.synchronized {
    if (localMode) {
      logger.info("Detect bigdl.localMode is set. Run workload without spark")
      // The physical core number should have been initialized
      // by java property -Dbigdl.coreNumber=xx
      setNodeAndCore(1, getCoreNumberFromProperty)
    } else {
      logger.info("Auto detect executor number and executor cores number")
      val (nExecutor, executorCores) = sparkExecutorAndCore().get
      logger.info(s"Executor number is $nExecutor and executor cores number is $executorCores")
      setNodeAndCore(nExecutor, executorCores)
      checkSparkContext
    }
  }

  private val logger = Logger.getLogger(getClass)
  private val singletonCounter = new AtomicBoolean()
  private var physicalCoreNumber = -1
  private var nodeNum: Int = -1

  private[bigdl] def localMode: Boolean = {
    System.getProperty("bigdl.localMode", "false").toLowerCase(Locale.ROOT) match {
      case "true" => true
      case "false" => false
      case option => throw new IllegalArgumentException(s"Unknown bigdl.localMode $option")
    }
  }

  private val NOT_INIT_ERROR =
    "Do you call Engine.init? See more at " +
      "https://github.com/intel-analytics/BigDL/wiki/Programming-Guide#engine"

  private val SPARK_CONF_ERROR = "For details please check " +
    "https://github.com/intel-analytics/BigDL/wiki/Programming-Guide#engine"

  /**
   * Notice: Please use property bigdl.engineType to set engineType.
   * Default engine is mklblas
   */
  private var engineType: EngineType = {
    System.getProperty("bigdl.engineType", "mklblas").toLowerCase(Locale.ROOT) match {
      case "mklblas" => MklBlas
      case engineType => throw new IllegalArgumentException(s"Unknown engine type $engineType")
    }
  }

  // Thread pool for default use
  @volatile private var _default: ThreadPool = null

  // Thread pool for layer use
  @volatile private var _model: ThreadPool = new ThreadPool(1).setMKLThread(MKL.getNumThreads)

  /**
   * If user undefine the property bigdl.coreNumber, it will return physical core number
   * system has. The biggest number it supports is the physical cores number.
   *
   * Currently, it not support detect true physical cores number. Get it through
   * Runtime.getRuntime().availableProcessors() / 2
   */
  private def getCoreNumberFromProperty() = {
    val coreNumber = System.getProperty("bigdl.coreNumber", getNumMachineCores.toString).toInt
    if (coreNumber > getNumMachineCores) {
      getNumMachineCores
    } else {
      coreNumber
    }
  }

  private def getNumMachineCores: Int = {
    // We assume the HT is enabled
    // Todo: check the Hyper threading
    Runtime.getRuntime().availableProcessors() / 2
  }

  /**
   * @return true if current execution is a singleton on the JVM
   */
  private[bigdl] def checkSingleton(): Boolean = singletonCounter.compareAndSet(false, true)

  /**
   * Reset the singleton flag
   */
  private[bigdl] def resetSingletonFlag(): Unit = singletonCounter.set(false)

  /**
   * Return number of cores, the engine.init must be called before use this method or an exception
   * will be thrown
   *
   * @return
   */
  private[bigdl] def coreNumber(): Int = {
    require(physicalCoreNumber != -1, s"Engine.init: Core number is " +
      s"not initialized. $NOT_INIT_ERROR")
    physicalCoreNumber
  }

  /**
   * This method should only be used for test purpose.
   *
   * @param n
   */
  private[bigdl] def setCoreNumber(n: Int): Unit = {
    require(n > 0, "Engine.init: core number is smaller than zero")
    physicalCoreNumber = n
    initThreadPool(n)
  }

  /**
   * Return node number, the engine.init must be called before use this method or an
   * exception will be thrown
   *
   * @return
   */
  private[bigdl] def nodeNumber(): Int = {
    require(nodeNum != -1, s"Engine.init: Node number is not initialized. $NOT_INIT_ERROR")
    nodeNum
  }

  /**
   * This method should only be used for test purpose.
   *
   * @param n
   */
  private[bigdl] def setNodeNumber(n : Int): Unit = {
    require(n > 0)
    nodeNum = n
  }

  /**
   * This method should only be used for test purpose.
   *
   * @param engineType
   */
  private[bigdl] def setEngineType(engineType: EngineType): Unit = {
    this.engineType = engineType
  }

  private[bigdl] def getEngineType(): EngineType = {
    this.engineType
  }

  private[bigdl] def model: ThreadPool = {
    _model
  }

  private[bigdl] def default: ThreadPool = {
    if (_default == null) {
      throw new IllegalStateException(s"Engine.init: Thread engine is not " +
        s"initialized. $NOT_INIT_ERROR")
    }
    _default
  }

  private def initThreadPool(core : Int) : Unit = {
    val defaultPoolSize: Int = System.getProperty("bigdl.utils.Engine.defaultPoolSize",
      (core * 50).toString).toInt
    if(_default == null || _default.getPoolSize != defaultPoolSize) {
      _default = new ThreadPool(defaultPoolSize)
    }

    val modelPoolSize: Int = if (engineType == MklBlas) {
      1
    } else {
      core
    }

    if(_model == null || _model.getPoolSize != modelPoolSize) {
      _model = new ThreadPool(modelPoolSize)
      _model.setMKLThread(MKL.getNumThreads)
    }
  }

  /**
   * Read conf values from config file
   * @return
   */
  private[utils] def readConf : Seq[(String, String)] = {
    val stream : InputStream = getClass.getResourceAsStream("/spark-bigdl.conf")
    val lines = scala.io.Source.fromInputStream(stream)
      .getLines.filter(_.startsWith("spark")).toArray
    lines.map(_.split("\\s+")).map(d => (d(0), d(1))).toSeq
  }

  /**
   * Check the spark conf of spark context if there's an exsiting one
   */
  private def checkSparkContext : Unit = {
    val tmpContext = SparkContext.getOrCreate(new SparkConf()
      .set("bigdl.temp.context", "true").setAppName("tmp context for Engine check"))
    // If there's already a spark context, it should not include the property
    val existingSparkContext = !tmpContext.getConf.contains("bigdl.temp.context")
    if (!existingSparkContext) {
      tmpContext.stop()
      throw new IllegalArgumentException("Engine.init: Cannot find an existing"
        + " spark context. Do you call this method after create spark context?")
    }
    logger.info("Find existing spark context. Checking the spark conf...")
    val sparkConf = tmpContext.getConf

    def verify(key: String, value: String): Unit = {
      val v = sparkConf.getOption(key)
      require(v.isDefined,
        s"Engine.init: Can not find $key. " + SPARK_CONF_ERROR)
      require(v.get == value,
        s"Engine.init: $key should be $value, " +
          s"but it is ${v.get}. " + SPARK_CONF_ERROR
      )
    }

    readConf.foreach(c => verify(c._1, c._2))
  }

  /**
   * Set executor number and cores per executor
   *
   * @param nodeNum
   * @param coreNum
   */
  private[bigdl] def setNodeAndCore(nodeNum: Int, coreNum: Int): Unit = {
    setNodeNumber(nodeNum)
    setCoreNumber(coreNum)
  }

  /**
   * Reset engine envs. Test purpose
   */
  private[bigdl] def reset : Unit = {
    nodeNum = 1
    physicalCoreNumber = 1
  }

  private def dynamicAllocationExecutor(conf: SparkConf): Option[Int] = {
    if (conf.get("spark.dynamicAllocation.enabled", null) == "true") {
      val maxExecutors = conf.get("spark.dynamicAllocation.maxExecutors", "1").toInt
      val minExecutors = conf.get("spark.dynamicAllocation.minExecutors", "1").toInt
      require(maxExecutors == minExecutors, "Engine.init: " +
        "spark.dynamicAllocation.maxExecutors and " +
        "spark.dynamicAllocation.minExecutors must be identical " +
        "in dynamic allocation for BigDL")
      Some(minExecutors)
    } else {
      None
    }
  }

  /**
   * Extract spark executor number and executor cores from environment.
   * @return (nExecutor, executorCore)
   */
  private[utils] def sparkExecutorAndCore(): Option[(Int, Int)] = {
    parseExecutorAndCore(SparkContext.getOrCreate().getConf)
  }

  /**
   * Extract spark executor number and executor cores from given conf.
   * Exposed for testing.
   * @return (nExecutor, executorCore)
   */
  private[utils] def parseExecutorAndCore(conf: SparkConf): Option[(Int, Int)] = {
    val master = conf.get("spark.master", null)
    if (master.toLowerCase.startsWith("local")) {
      // Spark local mode
      val patternLocalN = "local\\[(\\d+)\\]".r
      val patternLocalStar = "local\\[\\*\\]".r
      master match {
        case patternLocalN(n) => Some(1, n.toInt)
        case patternLocalStar(_*) => Some(1, getNumMachineCores)
        case _ => throw new IllegalArgumentException(s"Can't parser master $master")
      }
    } else if (master.toLowerCase.startsWith("spark")) {
      // Spark standalone mode
      val coreString = conf.get("spark.executor.cores", null)
      val maxString = conf.get("spark.cores.max", null)
      require(coreString != null, "Engine.init: Can't find executor core number" +
        ", do you submit with --executor-cores option")
      require(maxString != null, "Engine.init: Can't find total core number" +
        ". Do you submit with --total-executor-cores")
      val core = coreString.toInt
      val nodeNum = dynamicAllocationExecutor(conf).getOrElse {
        val total = maxString.toInt
        require(total >= core && total % core == 0, s"Engine.init: total core " +
          s"number($total) can't be divided " +
          s"by single core number($core) provided to spark-submit")
        total / core
      }
      Some(nodeNum, core)
    } else if (master.toLowerCase.startsWith("yarn")) {
      // yarn mode
      val coreString = conf.get("spark.executor.cores", null)
      require(coreString != null, "Engine.init: Can't find executor core number" +
        ", do you submit with " +
        "--executor-cores option")
      val core = coreString.toInt
      val node = dynamicAllocationExecutor(conf).getOrElse {
        val numExecutorString = conf.get("spark.executor.instances", null)
        require(numExecutorString != null, "Engine.init: Can't find executor number" +
          ", do you submit with " +
          "--num-executors option")
        numExecutorString.toInt
      }
      Some(node, core)
    } else if (master.toLowerCase.startsWith("mesos")) {
      // mesos mode
      require(conf.get("spark.mesos.coarse", null) != "false", "Engine.init: " +
        "Don't support mesos fine-grained mode")
      val coreString = conf.get("spark.executor.cores", null)
      require(coreString != null, "Engine.init: Can't find executor core number" +
        ", do you submit with --executor-cores option")
      val core = coreString.toInt
      val nodeNum = dynamicAllocationExecutor(conf).getOrElse {
        val maxString = conf.get("spark.cores.max", null)
        require(maxString != null, "Engine.init: Can't find total core number" +
          ". Do you submit with --total-executor-cores")
        val total = maxString.toInt
        require(total >= core && total % core == 0, s"Engine.init: total core " +
          s"number($total) can't be divided " +
          s"by single core number($core) provided to spark-submit")
        total / core
      }
      Some(nodeNum, core)
    } else {
      throw new IllegalArgumentException(s"Engine.init: Unsupported master format $master")
    }
  }
}
