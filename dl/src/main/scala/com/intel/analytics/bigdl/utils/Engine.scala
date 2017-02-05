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

import java.util.concurrent._
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.mkl.MKL
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.log4j.Logger
import org.apache.spark.SparkConf

import scala.collection.JavaConverters._
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}

sealed trait EngineType

case object MklBlas extends EngineType

/**
 * A thread pool wrapper, provide some helper functions for multi-threading
 */
class ThreadPool(private var poolSize: Int) {

  import ThreadPool._


  private var mklPoolSize : Option[Int] = None
  private var threadPool: ExecutorService = null

  private var context = spawnThreadPool(poolSize)

  private def spawnThreadPool(poolSize: Int): ExecutionContext = {
    if (poolSize == 1) {
      singleThreadPool
    } else {
      new ExecutionContext {
        if (threadPool != null) threadPool.shutdown()
        threadPool = Executors.newFixedThreadPool(poolSize, new ThreadFactory {
          override def newThread(r: Runnable): Thread = {
            val t = Executors.defaultThreadFactory().newThread(r)
            t.setDaemon(true)
            t
          }
        })

        def execute(runnable: Runnable) {
          threadPool.submit(runnable)
        }

        def reportFailure(t: Throwable) {}
      }
    }
  }

  def getPoolSize : Int = poolSize

  /**
   * Set MKL thread pool size
   *
   * @param size
   * @return
   */
  def setMKLThread(size: Int): this.type = {
    require(MKL.isMKLLoaded)
    mklPoolSize = Some(size)
    (1 to poolSize).map(i => Future {
      MKL.setNumThreads(size)
      val tid = Thread.currentThread().getId()
      logger.info(s"Set mkl threads to $size on thread $tid")
    }(context)).foreach(Await.result(_, Duration.Inf))
    this
  }

  /**
   * Invoke a batch of tasks and wait for all them finished
   *
   * @param tasks
   * @param timeout
   * @tparam T
   * @return
   */
  def invokeAndWait[T](tasks: Seq[() => T], timeout: Duration = Duration.Inf): Seq[T] = {
    tasks.map(task => Future {
      try {
        task()
      } catch {
        case t : Throwable =>
            logger.error("Error: " + ExceptionUtils.getStackTrace(t))
            throw t
      }
    }(context)).map(future => {
      Await.result(future, timeout)
    })
  }

  def invokeAndWait2[T](tasks: Seq[() => T], timeout: Long = Long.MaxValue,
    timeUnit: TimeUnit = TimeUnit.NANOSECONDS):
    scala.collection.mutable.Buffer[java.util.concurrent.Future[T]] = {
    val callables = tasks.map(task => new Callable[T] {
      override def call(): T = {
        try {
          task()
        } catch {
          case t : Throwable =>
            logger.error("Error: " + ExceptionUtils.getStackTrace(t))
            throw t
        }
      }
    })
    threadPool.invokeAll(callables.asJava, timeout, timeUnit).asScala
  }

  def invoke2[T](tasks: Seq[() => T]): Seq[java.util.concurrent.Future[T]] = {
    tasks.map(task => new Callable[T] {
      override def call(): T = {
        task()
      }
    }).map(threadPool.submit(_))
  }

  /**
   * Invoke a batch of tasks
   *
   * @param tasks
   */
  def invoke[T](tasks: Seq[() => T]): Seq[Future[T]] = {
    tasks.map(task => Future {
      try {
        task()
      } catch {
        case t : Throwable =>
          logger.error("Error: " + ExceptionUtils.getStackTrace(t))
          throw t
      }
    }(context))
  }

  /**
   * Invoke a single tasks
   *
   * @param task
   */
  def invoke[T](task: () => T): Future[T] = {
    Future {
      task()
    }(context)
  }

  /**
   * Wait for all the tasks in the wait queue finish
   *
   * @param timeout
   */
  def sync(futures: Seq[Future[_]], timeout: Duration = Duration.Inf): Unit = {
    futures.foreach(f => {
      Await.result(f, timeout)
    })
  }

  /**
   * Set pool size
   *
   * @param size
   * @return
   */
  def setPoolSize(size: Int): this.type = {
    if (size != poolSize) {
      context = spawnThreadPool(size)
      poolSize = size
      if(mklPoolSize.isDefined) {
        this.setMKLThread(mklPoolSize.get)
      }
    }
    this
  }
}

object ThreadPool {
  val singleThreadPool = new ExecutionContext {
    def execute(runnable: Runnable) {
      runnable.run()
    }

    def reportFailure(t: Throwable) {}
  }

  private val logger = Logger.getLogger(getClass)
}

object Engine {
  private val logger = Logger.getLogger(getClass)
  private val singletonCounter: AtomicInteger = new AtomicInteger(0)
  private[this] var _isInitialized: Boolean = false
  private[this] var _onSpark: Boolean = false

  def isInitialized: Boolean = _isInitialized

  def onSpark: Boolean = _onSpark

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

  def coreNumber(): Int = physicalCoreNumber

  /**
   * This method should only be used for test purpose.
   *
   * @param n
   */
  private[bigdl] def setCoreNumber(n: Int): Unit = {
    require(n > 0)
    physicalCoreNumber = n
    _model = initModelThreadPool()
  }

  // Set node number
  private var nodeNum: Int = if (System.getenv("DL_NODE_NUMBER") == null) {
    1
  } else {
    System.getenv("DL_NODE_NUMBER").toInt
  }

  def nodeNumber(): Int = nodeNum

  /**
   * This method should only be used for test purpose.
   *
   * @param n
   */
  private[bigdl] def setNodeNumber(n : Int): Unit = {
    nodeNum = n
  }

  private val ERROR = "Please use bigdl.sh set the env. For spark application, please use " +
    "Engine.sparkConf() to initialize your sparkConf"

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

  private val defaultPoolSize: Int = System.getProperty("bigdl.utils.Engine.defaultPoolSize",
    (physicalCoreNumber * 50).toString).toInt

  val default: ThreadPool = new ThreadPool(defaultPoolSize)

  @volatile private var _model: ThreadPool = initModelThreadPool()

  def model: ThreadPool = _model

  private def initModelThreadPool() = {
    val modelPoolSize: Int = if (engineType == MklBlas) {
      1
    } else {
      physicalCoreNumber
    }

    val model = new ThreadPool(modelPoolSize)
    model.setMKLThread(1)
    model
  }

  @deprecated
  def init(
    node: Int,
    cores: Int,
    onSpark: Boolean = false
  ): Option[SparkConf] = {
    val result = init
    require(node == Engine.nodeNumber, s"The input node number($node) is inconsistent with " +
      s"auto-detect node number(${Engine.nodeNumber})")
    require(cores == Engine.coreNumber, s"The input core number($cores) is inconsistent with " +
      s"auto-detect core number(${Engine.coreNumber})")
    result
  }

  /**
   * Try to automatically find node number and core number from the environment.
   *
   * We assume that user run their spark application through spark-submit. And try to find the node
   * number and core number from the system property set by spark-submit. Application should use
   * this method to get SparkConf object to init their SparkContext.
   *
   * If application is not submitted by spark-submit, we consider it run on a single node without
   * Spark. Note that this is different from Spark Local mode. If you want to run in Spark Local
   * mode, you still need to submit your application through spark-submit --master local[n].
   *
   * @return An option of SparkConf
   */
  def init: Option[SparkConf] = {
    val ret = if (System.getProperty("SPARK_SUBMIT") != null) {
      _onSpark = true
      val (node, cores) = sparkExecutorAndCore
      nodeNum = node
      physicalCoreNumber = cores
      _model = initModelThreadPool()
      val sc = if (engineType == MklBlas) {
        new SparkConf()
          .setExecutorEnv("DL_ENGINE_TYPE", "mklblas")
          .setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
          .setExecutorEnv("KMP_BLOCKTIME", "0")
          .setExecutorEnv("OMP_WAIT_POLICY", "passive")
          .setExecutorEnv("OMP_NUM_THREADS", "1")
          .setExecutorEnv("DL_CORE_NUMBER", coreNumber.toString)
          .setExecutorEnv("DL_NODE_NUMBER", nodeNum.toString)
          .set("spark.shuffle.reduceLocality.enabled", "false")
          .set("spark.shuffle.blockTransferService", "nio")
          .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")
      } else {
        throw new IllegalArgumentException(engineType.toString)
      }
      Some(sc)
    } else {
      _onSpark = false
      None
    }
    _isInitialized = true

    ret
  }

  /**
   * Reset engine envs. Test purpose
   */
  private[bigdl] def reset : Unit = {
    _onSpark = false
    _isInitialized = false
    nodeNum = 1
    physicalCoreNumber = 1
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

  private def sparkExecutorAndCore : (Int, Int) = {
    require(System.getProperty("spark.master") != null, "Can't find spark.master, do you start " +
      "your application without spark-submit?")
    val master = System.getProperty("spark.master")
    if(master.toLowerCase.startsWith("local")) {
      // Spark local mode
      val pattern = "local\\[(\\d+)\\]".r
      master match {
        case pattern(n) => (1, n.toInt)
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
      (nodeNum, core)
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
      (node, core)
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
      (nodeNum, core)
    } else {
      throw new IllegalArgumentException(s"Unsupported master format $master")
    }
  }

  // Check envs
  if (Engine.getEngineType() == MklBlas) {
    if (System.getenv("OMP_NUM_THREADS") != "1"
      || System.getenv("OMP_WAIT_POLICY") != "passive"
      || System.getenv("KMP_BLOCKTIME") != "0") {
      logger.warn("Invalid env setting. " + ERROR)
    }
  } else {
    throw new IllegalArgumentException(engineType.toString)
  }
}
