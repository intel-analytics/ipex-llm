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

import java.io.{FileOutputStream, InputStream, PrintWriter}
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

import org.apache.log4j.Logger
import org.apache.spark._
import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.mkl.hardware.{Affinity, CpuInfo}
import org.apache.spark.utils.SparkUtils
import py4j.GatewayServer

import scala.util.control.{ControlThrowable, NonFatal}

/**
 * define engine type trait
 */
sealed trait EngineType

case object MklBlas extends EngineType
case object MklDnn extends EngineType

/**
 * define optimizer version trait
 */
sealed trait OptimizerVersion

case object OptimizerV1 extends OptimizerVersion
case object OptimizerV2 extends OptimizerVersion


object Engine {

  // Initialize some properties for mkldnn engine. We should call it at the beginning.
  // Otherwise some properties will have no effect.
  if (System.getProperty("bigdl.engineType") == "mkldnn" &&
    System.getProperty("bigdl.multiModels", "false") == "false") {
    setMklDnnEnvironments()
  }

  @deprecated(
    "See https://bigdl-project.github.io/master/#APIGuide/Engine/",
    "0.1.0")
  def init(nExecutor: Int,
           executorCores: Int,
           onSpark: Boolean): Option[SparkConf] = {
    logger.warn("Engine.init(nExecutor, executorCores, onSpark) is deprecated. " +
      "Please refer to " +
      "https://bigdl-project.github.io/master/#APIGuide/Engine/")
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

  @volatile
  private var gatewayServer: py4j.GatewayServer = null

  private def createGatewayPortFile(port: Int): Unit = {
    val file = new java.io.File(SparkFiles.getRootDirectory(), "gateway_port")
    logger.debug(s"Creating JavaGatewayServer port file" +
      s" on executor-${SparkEnv.get.executorId}:${file.getAbsolutePath}")
    if (file.exists()) {
      file.delete()
    }
    file.createNewFile()
    val out = new PrintWriter(file)
    try {
      out.print(port)
      out.flush()
    } finally {
      out.close()
    }
  }

  private[bigdl] def createJavaGateway(driverPort: Int): Unit = {
    if (gatewayServer != null) return
    this.synchronized {
      if (gatewayServer != null) return
      gatewayServer = new py4j.GatewayServer(null, 0)
    }

    logger.info(s"Initializing JavaGatewayServer on executor-${SparkEnv.get.executorId} ")
    GatewayServer.turnLoggingOn()
    val thread = new Thread(new Runnable() {
      override def run(): Unit = try {
        gatewayServer.start()
      } catch {
        case ct: ControlThrowable =>
          throw ct
        case t: Throwable =>
          throw new Exception(s"Uncaught exception " +
            s"in thread ${Thread.currentThread().getName}, when staring JavaGatewayServer", t)
      }
    })
    thread.setName("py4j-executor-gateway-init")
    thread.setDaemon(true)
    thread.start()

    thread.join()

    logger.info(s"JavaGatewayServer initialized")

    Runtime.getRuntime().addShutdownHook(new Thread {
      override def run(): Unit = {
        gatewayServer.shutdown()
      }
    })

    try {
      createGatewayPortFile(gatewayServer.getListeningPort)
    } catch {
      case NonFatal(e) =>
        throw new Exception("Could not create java gateway port file", e)
    }
  }




  private[bigdl] def localMode: Boolean = {
    System.getProperty("bigdl.localMode", "false").toLowerCase(Locale.ROOT) match {
      case "true" => true
      case "false" => false
      case option => throw new IllegalArgumentException(s"Unknown bigdl.localMode $option")
    }
  }

  private val NOT_INIT_ERROR =
    "Do you call Engine.init? See more at " +
      "https://bigdl-project.github.io/master/#APIGuide/Engine/"

  private val SPARK_CONF_ERROR = "For details please check " +
    "https://bigdl-project.github.io/master/#APIGuide/Engine/"

  /**
   * Notice: Please use property bigdl.engineType to set engineType.
   * Default engine is mklblas
   */
  private var engineType: EngineType = {
    System.getProperty("bigdl.engineType", "mklblas").toLowerCase(Locale.ROOT) match {
      case "mklblas" => MklBlas
      case "mkldnn" => MklDnn
      case engineType => throw new IllegalArgumentException(s"Unknown engine type $engineType")
    }
  }

  /**
   * Notice: Please use property bigdl.optimizerVersion to set optimizerVersion.
   * Default version is OptimizerV1
   */
  private var optimizerVersion: OptimizerVersion = {
    System.getProperty("bigdl.optimizerVersion", "optimizerv1").toLowerCase(Locale.ROOT) match {
      case "optimizerv1" => OptimizerV1
      case "optimizerv2" => OptimizerV2
      case optimizerVersion => throw new IllegalArgumentException(s"Unknown type $optimizerVersion")
    }
  }

  // Thread pool for default use
  @volatile private var _default: ThreadPool = null

  // Thread pool for layer use
  @volatile private var _model: ThreadPool = new ThreadPool(1)

  // Thread pool for blas wrapper layer
  private[bigdl] var wrapperComputing: ThreadPool = null

  // This thread is mainly for mkldnn library.
  // Because if we use the parent thread directly, there will be two bugs,
  //   1. The child threads forked from parent thread will be bound to core 0
  //      because of the affinity settings.
  //   2. The native thread has some unknown thread local variables. So if
  //      the parent thread exits and is recreated, such as the thread from
  //      Executors.newFixedThreadPool. The whole app will be segment fault.
  // The parent thread means the main thread (Local Mode) or worker thread of
  // `mapPartition` (Distributed Mode).
  // --------------------------------------------------------------------------
  // We will only use the `threadPool` in ThreadPool, which is a ExecutorService.
  // For `context` in ThreadPool, it is the called thread when poolSize is 1.
  // So many usages of that thread, we will not change it for now.
  val dnnComputing: ThreadPool = new ThreadPool(1)
  // We need to init dnn thread in case that users directly call model operation in java local
  initDnnThread()

  /**
   * If user undefine the property bigdl.coreNumber, it will return physical core number
   * system has. The biggest number it supports is the physical cores number.
   *
   * Currently, it not support detect true physical cores number. Get it through
   * Runtime.getRuntime().availableProcessors() / 2
   */
  private def getCoreNumberFromProperty() = {
    System.getProperty("bigdl.coreNumber", getNumMachineCores.toString).toInt
  }

  private def getNumMachineCores: Int = {
    val coreNum = Runtime.getRuntime().availableProcessors()
    require(coreNum > 0, "Get a non-positive core number")
    // We assume the HT is enabled
    // TODO: check the Hyper threading
    if (coreNum > 1) coreNum / 2 else 1
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
   * @param optimizerVersion
   */
  private[bigdl] def setOptimizerVersion(optimizerVersion : OptimizerVersion): Unit = {
    this.optimizerVersion = optimizerVersion
  }

  private[bigdl] def getOptimizerVersion(): OptimizerVersion = {
    this.optimizerVersion
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

  private[bigdl] def isMultiModels: Boolean = {
    getEngineType() match {
      case MklBlas => true
      case MklDnn => System.getProperty("bigdl.multiModels", "false").toBoolean
    }
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
    if (wrapperComputing == null || wrapperComputing.getPoolSize != defaultPoolSize) {
      wrapperComputing = new ThreadPool(defaultPoolSize)
    }

    // for dnn model we should set the pool size to 1 also.
    // otherwise, it will downgrade the performance and
    // FIXME make the loss to NaN.
    val modelPoolSize = 1

    if(_model == null || _model.getPoolSize != modelPoolSize) {
      _model = new ThreadPool(modelPoolSize)
    }
    _model.setMKLThread(MKL.getMklNumThreads)

    // do two things, set number of threads for omp thread pool and set the affinity
    // only effects the `threadPool` and `computing.invoke/invokeAndWait` will not
    // be effected. And affinity will not effect the other threads except
    // this thread and the omp threads forked from computing.
    if (engineType == MklDnn) {
      dnnComputing.setMKLThreadOfMklDnnBackend(MKL.getMklNumThreads)
      _model.setMKLThreadOfMklDnnBackend(MKL.getMklNumThreads)
    }
    if (System.getProperty("multiThread", "false").toBoolean) {
      wrapperComputing.setMKLThread(1)
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

    // For spark 1.5, we observe nio block manager has better performance than netty block manager
    // So we will force set block manager to nio. If user don't want this, he/she can set
    // bigdl.network.nio == false to customize it. This configuration/blcok manager setting won't
    // take affect on newer spark version as the nio block manger has been removed
    lines.map(_.split("\\s+")).map(d => (d(0), d(1))).toSeq
      .filter(_._1 != "spark.shuffle.blockTransferService" ||
        System.getProperty("bigdl.network.nio", "true").toBoolean)
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
    try {
      parseExecutorAndCore(SparkContext.getOrCreate().getConf)
    } catch {
      case s: SparkException =>
        if (s.getMessage.contains("A master URL must be set in your configuration")) {
          throw new IllegalArgumentException("A master URL must be set in your configuration." +
            " Or if you want to run BigDL in a local JVM environment, you should set Java " +
            "property bigdl.localMode=true")
        }
        throw s
    }
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
    } else if (master.toLowerCase.startsWith("k8s")) {
      // Spark-on-kubernetes mode
      val coreString = conf.get("spark.executor.cores", null)
      val maxString = conf.get("spark.cores.max", null)
      require(coreString != null, "Engine.init: Can't find executor core number" +
        ", do you submit with --conf spark.executor.cores option")
      require(maxString != null, "Engine.init: Can't find total core number" +
        ". Do you submit with --conf spark.cores.max option")
      val core = coreString.toInt
      val nodeNum = dynamicAllocationExecutor(conf).getOrElse {
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

  private def setMklDnnEnvironments(): Unit = {
    import com.intel.analytics.bigdl.mkl.hardware.CpuInfo
    val affinityCores = Affinity.getAffinity
    val physicalCoreNum = CpuInfo.getPhysicalProcessorCount
    val affinityCoreNum = affinityCores.length

    // 1. this library in docker/cgroup env, which sets cpu affinity fist. so we can't use
    //    resources exceeding limits.
    // 2. this library is in a hyper threading envs, so we should set the mkl num threads
    //    to physical core number for performance

    val default = if (affinityCores.min > 0 && affinityCores.max >= physicalCoreNumber) {
      affinityCoreNum
    } else if (physicalCoreNum > affinityCoreNum ) {
      affinityCoreNum
    } else {
      physicalCoreNum
    }

    val threadsNumber = System.getProperty("bigdl.mklNumThreads", default.toString)
    System.setProperty("bigdl.mklNumThreads", s"$threadsNumber")

    System.setProperty("bigdl.disable.mklBlockTime", "true")
    System.setProperty("bigdl.coreNumber", "1")
  }

  private def initDnnThread(): Unit = {
    if (engineType == MklDnn) {
      dnnComputing.setMKLThreadOfMklDnnBackend(MKL.getMklNumThreads)
    }
  }
}
