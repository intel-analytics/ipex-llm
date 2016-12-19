/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{ConcurrentLinkedQueue, Executors, ThreadFactory}

import com.intel.analytics.bigdl.mkl.MKL
import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, Logging}

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}

sealed trait EngineType

case object MklBlas extends EngineType

case object MklDnn extends EngineType

/**
 * A thread pool wrapper, provide some helper functions for multi-threading
 */
class ThreadPool(private var poolSize: Int) {

  import ThreadPool._

  private var context = spawnThreadPool(poolSize)

  private def spawnThreadPool(poolSize: Int): ExecutionContext = {
    if (poolSize == 1) {
      singleThreadPool
    } else {
      new ExecutionContext {
        val threadPool = Executors.newFixedThreadPool(poolSize, new ThreadFactory {
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

  /**
   * Set MKL thread pool size
   *
   * @param size
   * @return
   */
  def setMKLThread(size: Int): this.type = {
    require(MKL.isMKLLoaded)
    (1 to poolSize).map(i => Future {
      MKL.setNumThreads(size)
      val tid = Thread.currentThread().getId()
      logger.info(s"Set mkl threads to 1 on thread $tid")
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
      task()
    }(context)).map(future => Await.result(future, timeout))
  }

  /**
   * Invoke a batch of tasks
   *
   * @param tasks
   */
  def invoke[T](tasks: Seq[() => T]): Seq[Future[T]] = {
    tasks.map(task => Future {
      task()
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

/**
 * Mange thread parallel behavior
 */
object Engine {
  private val logger = Logger.getLogger(getClass)

  private val singletonCounter: AtomicInteger = new AtomicInteger(0)

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

  // Set node number
  private var nodeNum: Option[Int] = None

  private val ERROR = "Please use bigdlvars.sh set the env. For spark application, please use" +
    "Engine.sparkConf() to initialize your sparkConf"

  /**
   * Notice: Please use property DL_ENGINE_TYPE to set engineType.
   * Default engine is MklBlas
   */
  private var engineType: EngineType = {
    val dlEngineType = System.getenv("DL_ENGINE_TYPE")

    if (dlEngineType == null || dlEngineType.toLowerCase == "mklblas") {
      MklBlas
    } else if (dlEngineType.toLowerCase == "mkldnn") {
      MklDnn
    } else {
      throw new Error(s"Unkown DL_ENGINE_TYPE. $ERROR")
    }
  }

  private val defaultPoolSize: Int = System.getProperty("bigdl.utils.Engine.defaultPoolSize",
    (Runtime.getRuntime().availableProcessors() / 2 * 50).toString).toInt

  val default: ThreadPool = new ThreadPool(defaultPoolSize)

  private var _model : ThreadPool = null

  def model : ThreadPool = {
    if (_model == null) {
      val modelPoolSize: Int = if (engineType == MklBlas) {
        1
      } else {
        physicalCoreNumber
      }

      _model = new ThreadPool(modelPoolSize)
      _model.setMKLThread(1)
    }
    _model
  }

  /**
   * Check if current execution is a singleton on the JVM
   *
   * @return
   */
  def checkSingleton(): Boolean = {
    val count = singletonCounter.incrementAndGet()
    (count == 1)
  }

  // Set spark envs
  def sparkConf(): SparkConf = {
    require(nodeNum.isDefined, "Please set node number and core number per node by Engine" +
      ".setCluster()")
    if (engineType == MklBlas) {
      new SparkConf().setExecutorEnv("DL_ENGINE_TYPE", "mklblas")
        .setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
        .setExecutorEnv("KMP_BLOCKTIME", "0")
        .setExecutorEnv("OMP_WAIT_POLICY", "passive")
        .setExecutorEnv("OMP_NUM_THREADS", "1")
        .setExecutorEnv("DL_CORE_NUMBER", coreNumber().toString)
        .set("spark.task.maxFailures", "1")
        .set("spark.shuffle.blockTransferService", "nio")
        .set("spark.akka.frameSize", "10")
    } else {
      new SparkConf().setExecutorEnv("DL_ENGINE_TYPE", "mkldnn")
        .setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
        .setExecutorEnv("DL_CORE_NUMBER", coreNumber().toString)
        .set("spark.task.maxFailures", "1")
        .set("spark.shuffle.blockTransferService", "nio")
        .set("spark.akka.frameSize", "10")
    }
  }

  private[bigdl] def setEngineType(engineType: EngineType): Unit = {
    this.engineType = engineType
  }

  def getEngineType(): EngineType = {
    this.engineType
  }

  // Check envs
  if (Engine.getEngineType() == MklBlas) {
    if (System.getenv("OMP_NUM_THREADS") != "1"
      || System.getenv("OMP_WAIT_POLICY") != "passive"
      || System.getenv("KMP_BLOCKTIME") != "0") {
      logger.warn("Invalid env setting. " + ERROR)
    }
  } else if (Engine.getEngineType() == MklDnn) {
    if (System.getenv("OMP_NUM_THREADS") != null
      || System.getenv("OMP_WAIT_POLICY") != null
      || System.getenv("KMP_BLOCKTIME") != null) {
      logger.warn("Invalid env setting. " + ERROR)
    }
  }

  def coreNumber(): Int = physicalCoreNumber

  def setCoreNumber(n: Int): Unit = {
    require(n > 0)
    physicalCoreNumber = n
  }

  def nodeNumber(): Option[Int] = nodeNum

  def setCluster(n: Int, c: Int): Unit = {
    require(n > 0)
    nodeNum = Some(n)
    physicalCoreNumber = c
  }
}
