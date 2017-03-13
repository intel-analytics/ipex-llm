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

import java.io.InputStream
import java.util.concurrent.atomic.AtomicInteger
import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext}

sealed trait EngineType

case object MklBlas extends EngineType


object Engine {
  private val logger = Logger.getLogger(getClass)
  private val singletonCounter: AtomicInteger = new AtomicInteger(0)

  // Detect this env means Engine.init is called
  private[this] var _isInitialized: Boolean = if (System.getenv("ON_SPARK") == null) {
    false
  } else {
    true
  }

  private[this] var _onSpark: Boolean = if (System.getenv("ON_SPARK") == null) {
    false
  } else {
    true
  }

  /**
   * If the engine is initialized
   *
   * @return
   */
  def isInitialized: Boolean = _isInitialized

  /**
   * If current JVM is a spark Executor
   *
   * @return
   */
  def onSpark: Boolean = _onSpark

  private var localMode: Boolean = {
    val env = System.getenv("LOCAL_MODE")
    if(env == null) {
      false
    } else {
      true
    }
  }

  private[utils] def setLocalMode = this.localMode = true

  /**
   * Check if current execution is a singleton on the JVM
   *
   * @return
   */
  def checkSingleton(): Boolean = {
    val count = singletonCounter.incrementAndGet()
    (count == 1)
  }

  /**
   * Reset the singleton flag
   */
  def resetSingletonFlag(): Unit = {
    singletonCounter.set(0)
  }

  private var physicalCoreNumber = {
    val env = System.getenv("DL_CORE_NUMBER")
    if(env == null) {
      // We assume the HT is enabled
      // Todo: check the Hyper threading
      Runtime.getRuntime().availableProcessors() / 2
    } else {
      env.toInt
    }
  }

  /**
   * Return number of cores, the engine.init must be called before use this method or an exception
   * will be thrown
   *
   * @return
   */
  def coreNumber(): Int = {
    if (!_isInitialized) throw new IllegalStateException("Engine is not inited")

    physicalCoreNumber
  }

  /**
   * This method should only be used for test purpose.
   *
   * @param n
   */
  private[bigdl] def setCoreNumber(n: Int): Unit = {
    require(n > 0)
    physicalCoreNumber = n
    initThreadPool(n)
  }

  private var nodeNum: Int = -1

  /**
   * Return node number, the engine.init must be called before use this method or an
   * exception will be thrown
   *
   * @return
   */
  def nodeNumber(): Int = {
    if (!_isInitialized) throw new IllegalStateException("Engine is not inited")
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

  private val ERROR = "Current environment variable looks not correct. Please use bigdl.sh to " +
    "start your application. For details, see " +
    "https://github.com/intel-analytics/BigDL/wiki/Getting-Started#before-running-a-bigdl-program"

  /**
   * Notice: Please use property DL_ENGINE_TYPE to set engineType.
   * Default engine is MklBlas
   */
  private var engineType: EngineType = {
    val dlEngineType = System.getenv("DL_ENGINE_TYPE")

    if (dlEngineType == null || dlEngineType.toLowerCase == "mklblas") {
      MklBlas
    } else {
      throw new Error(s"Unknown DL_ENGINE_TYPE. $ERROR")
    }
  }

  /**
   * This method should only be used for test purpose.
   *
   * @param engineType
   */
  private[bigdl] def setEngineType(engineType: EngineType): Unit = {
    this.engineType = engineType
  }

  def getEngineType(): EngineType = {
    this.engineType
  }

  @volatile private var _default: ThreadPool = null

  @volatile private var _model: ThreadPool = new ThreadPool(1).setMKLThread(1)

  def model: ThreadPool = {
    if (_model == null) {
      throw new IllegalStateException("Model engine is not initialized. Have you call Engine.init?")
    }
    _model
  }

  def default: ThreadPool = {
    if (_default == null) {
      throw new IllegalStateException("Thread engine is not initialized. Have you call Engine" +
        ".init?")
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
      _model.setMKLThread(1)
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

  private val errorMsg = "For details please check " +
    "https://github.com/intel-analytics/BigDL/wiki/Programming-Guide#engine"

  /**
   * Check the spark conf of spark context if there's an exsiting one
   */
  private def checkSparkContext : Unit = {
    val tmpContext = SparkContext.getOrCreate(new SparkConf().set("tmpContext", "true"))
    // If there's already a spark context, it should not include the property
    val exisitingSparkContext = !tmpContext.getConf.contains("tmpContext")
    require(exisitingSparkContext, "Cannot find an existing spark context. " +
      "Do you call this method after create spark context?")
    logger.info("Find existing spark context. Checking the spark conf...")
    val sparkConf = tmpContext.getConf

    def verify(key: String, value: String): Unit = {
      for ((k, v) <- sparkConf.getAll) {
        if (k == key) {
          if (value != v) {
            throw new IllegalArgumentException(s"$k should be $value, but it is $v. " + errorMsg)
          }
          return
        }
      }
      throw new IllegalArgumentException(s"Can not find $key. " + errorMsg)
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

  @deprecated
  def init(nExecutor: Int,
           executorCores: Int,
           onSpark: Boolean): Option[SparkConf] = {
    setNodeAndCore(nExecutor, executorCores)
    val res = if (onSpark) {
      _onSpark = onSpark
      Some(createSparkConf())
    } else {
      None
    }
    _isInitialized = true
    res
  }

  /**
    * BigDL need some spark conf values to be set correctly to have a better performance.
    *
    * This method will create a spark conf, or use existing one if you provided on.
    * Populate it with correct values.
    *
    * We recommand you use this method instead of setting spark conf values directly. This can the
    * spark conf values changes transparent to you. However, if you use spark-shell or
    * Jupiter notebook, as the spark context is created before your code, you have to
    * set them directly(through command line options or properties-file)
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
   * This method should be call before any BigDL procedure and after spark context created.
   *
   * BigDL need some spark conf values to be set correctly to have a better performance. There's
   * also multi-thread engines so executor number and core number per executor need to be know to
   * set the parameter of these engines correctly.
   *
   * The method can set parameters of multi-thread engines, verify spark conf values of an
   * existing spark context.
   */
  private[bigdl] def init: Unit = this.synchronized {
    if (localMode) {
      // The physical core number should have been initialized by env variable in bigdl.sh
      setNodeAndCore(1, physicalCoreNumber)
    } else {
      logger.info("Auto detect node number and cores number")
      val (nExecutor, executorCores) = sparkExecutorAndCore(forceCheck = true).get
      setNodeAndCore(nExecutor, executorCores)
      checkSparkContext
      _onSpark = true
    }
    _isInitialized = true
  }

  /**
   * Reset engine envs. Test purpose
   */
  private[bigdl] def reset : Unit = {
    _onSpark = false
    _isInitialized = false
    nodeNum = 1
    physicalCoreNumber = 1
    localMode = false
  }

  private def dynamicAllocationExecutor : Option[Int] = {
    if (System.getProperty("spark.dynamicAllocation.enabled") == "true") {
      val maxExecutors = if (System.getProperty("spark.dynamicAllocation.maxExecutors") != null) {
        System.getProperty("spark.dynamicAllocation.maxExecutors").toInt
      } else {
        1
      }
      val minExecutors = if (System.getProperty("spark.dynamicAllocation.minExecutors") != null) {
        System.getProperty("spark.dynamicAllocation.minExecutors").toInt
      } else {
        1
      }
      require(maxExecutors == minExecutors, "spark.dynamicAllocation.maxExecutors and " +
        "spark.dynamicAllocation.minExecutors must be identical in dynamic allocation for BigDL")
      Some(minExecutors)
    } else {
      None
    }
  }

  /**
   * Extract spark executor number and executor cores from environment.
    * @param forceCheck throw exception if user doesn't set properties correctly
   * @return (nExecutor, executorCore)
   */
  private[utils] def sparkExecutorAndCore(forceCheck : Boolean) : Option[(Int, Int)] = {
    val master = System.getProperty("spark.master")
    if (master == null) {
      require(forceCheck == false, "Can't find spark.master, do you start " +
        "your application with spark-submit?")
      return None
    }
    if(master.toLowerCase.startsWith("local")) {
      // Spark local mode
      val patternLocalN = "local\\[(\\d+)\\]".r
      val patternLocalStar = "local\\[\\*\\]".r
      master match {
        case patternLocalN(n) => Some(1, n.toInt)
        case patternLocalStar => Some(1, physicalCoreNumber)
        case _ => throw new IllegalArgumentException(s"Can't parser master $master")
      }
    } else if (master.toLowerCase.startsWith("spark")) {
      // Spark standalone mode
      val coreString = System.getProperty("spark.executor.cores")
      val maxString = System.getProperty("spark.cores.max")
      require(coreString != null, "Can't find executor core number, do you submit with " +
        "--executor-cores option")
      require(maxString != null, "Can't find total core number. Do you submit with " +
        "--total-executor-cores")
      val core = coreString.toInt
      val nodeNum = dynamicAllocationExecutor.getOrElse {
        val total = maxString.toInt
        require(total > core && total % core == 0, s"total core number($total) can't be divided " +
          s"by single core number($core) provided to spark-submit")
        total / core
      }
      Some(nodeNum, core)
    } else if (master.toLowerCase.startsWith("yarn")) {
      // yarn mode
      val coreString = System.getProperty("spark.executor.cores")
      require(coreString != null, "Can't find executor core number, do you submit with " +
        "--executor-cores option")
      val core = coreString.toInt
      val node = dynamicAllocationExecutor.getOrElse {
        val numExecutorString = System.getProperty("spark.executor.instances")
        require(numExecutorString != null, "Can't find executor number, do you submit with " +
          "--num-executors option")
        numExecutorString.toInt
      }
      Some(node, core)
    } else if (master.toLowerCase.startsWith("mesos")) {
      // mesos mode
      require(System.getProperty("spark.mesos.coarse") != "false", "Don't support mesos " +
        "fine-grained mode")
      val coreString = System.getProperty("spark.executor.cores")
      require(coreString != null, "Can't find executor core number, do you submit with " +
        "--executor-cores option")
      val core = coreString.toInt
      val nodeNum = dynamicAllocationExecutor.getOrElse {
        val maxString = System.getProperty("spark.cores.max")
        require(maxString != null, "Can't find total core number. Do you submit with " +
          "--total-executor-cores")
        val total = maxString.toInt
        require(total > core && total % core == 0, s"total core number($total) can't be divided " +
          s"by single core number($core) provided to spark-submit")
        total / core
      }
      Some(nodeNum, core)
    } else {
      throw new IllegalArgumentException(s"Unsupported master format $master")
    }
  }
}
