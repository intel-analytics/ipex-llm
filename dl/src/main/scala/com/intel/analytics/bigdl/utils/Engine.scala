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
 * Provide appropriated thread pool based on user provided parallelism
 */
object Engine{
  private val logger = Logger.getLogger(getClass)

  private val singletonCounter : AtomicInteger = new AtomicInteger(0)

  def checkSingleton() : Boolean = {
    val count = singletonCounter.incrementAndGet()
    (count == 1)
  }

  private val defaultPoolSize: Int =
    System.getProperty("com.intel.analytics.bigdl.utils.Engine.defaultPoolSize",
      (Runtime.getRuntime().availableProcessors() / 2).toString).toInt
  private val queuePointer = new AtomicInteger(0)
  private val maxQueueSize: Int =
    System.getProperty("com.intel.analytics.bigdl.utils.Engine.maxQueueSize", "500").toInt
  private val waitQueue : Array[Future[_]] = new Array[Future[_]](maxQueueSize)

  private val threadPool : ThreadLocal[ExecutionContext] = new ThreadLocal[ExecutionContext]

  private def spawnThreadPool(poolSize : Int) : ExecutionContext = {
    new ExecutionContext {
      val pool = Executors.newFixedThreadPool(poolSize, new ThreadFactory {
        override def newThread(r: Runnable): Thread = {
          val t = Executors.defaultThreadFactory().newThread(r)
          t.setDaemon(true)
          t
        }
      })

      def execute(runnable: Runnable) {
        pool.submit(runnable)
      }

      def reportFailure(t: Throwable) {}
    }
  }

  def setThreadPool(size : Int): Unit = {
    threadPool.set(spawnThreadPool(defaultPoolSize))
  }

  def setThreadPool(): Unit = setThreadPool(defaultPoolSize)

  private def getThreadPool(): ExecutionContext = {
    var pool = threadPool.get()
    if(pool == null) {
      setThreadPool()
      pool = threadPool.get()
    }
    pool
  }

  def invokeAndWait[T](tasks: Seq[() => T], timeout : Duration = Duration.Inf) : Seq[T] = {
    tasks.map(task => Future {
      task()
    }(getThreadPool)).map(future => Await.result(future, timeout))
  }

  def invoke(tasks: Seq[() => _]) : Unit = {
    tasks.map(task => Future {
      task()
    }(getThreadPool)).foreach(f => {
      val i = queuePointer.getAndIncrement()
      require(i < maxQueueSize, "Queue is full. Please consider increase waiting queue size")
      waitQueue(i) = f
    })
  }

  def sync(timeout : Duration = Duration.Inf) : Unit = {
    val length = queuePointer.get()
    var i = 0
    while(i < length) {
      Await.result(waitQueue(i), timeout)
      i += 1
    }
    queuePointer.set(0)
  }

  private val ERROR = "Please use bigdlvars.sh set the env. For spark application, please use" +
    "Engine.sparkConf() to initialize your sparkConf"

  /**
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

  def sparkConf(): SparkConf = {
    if(engineType == MklBlas) {
      new SparkConf().setExecutorEnv("DL_ENGINE_TYPE", "mklblas")
        .setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
        .setExecutorEnv("KMP_BLOCKTIME", "0")
        .setExecutorEnv("OMP_WAIT_POLICY", "passive")
        .setExecutorEnv("OMP_NUM_THREADS", "1")
        .set("spark.task.maxFailures", "1")
        .set("spark.shuffle.blockTransferService", "nio")
        .set("spark.akka.frameSize", "10")
        .set("spark.task.cpus", "28")
    } else {
      new SparkConf().setExecutorEnv("DL_ENGINE_TYPE", "mkldnn")
        .setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
        .set("spark.task.maxFailures", "1")
        .set("spark.shuffle.blockTransferService", "nio")
        .set("spark.akka.frameSize", "10")
        .set("spark.task.cpus", "28")
    }
  }

  /**
   * Notice: Please use property DL_ENGINE_TYPE to set engineType.
   */
  private[bigdl] def setEngineType(engineType: EngineType) : Unit = {
    this.engineType = engineType
  }

  def getEngineType() : EngineType = {
    this.engineType
  }

  if(Engine.getEngineType() == MklBlas) {
    if(System.getenv("OMP_NUM_THREADS") != "1"
      || System.getenv("OMP_WAIT_POLICY") != "passive"
      || System.getenv("KMP_BLOCKTIME") != "0") {
      logger.warn("Invalid env setting. " + ERROR)
    }
  } else if(Engine.getEngineType() == MklDnn){
    if(System.getenv("OMP_NUM_THREADS") != null
      || System.getenv("OMP_WAIT_POLICY") != null
      || System.getenv("KMP_BLOCKTIME") != null) {
      logger.warn("Invalid env setting. " + ERROR)
    }
  }

  // We assume the HT is enble
  // Todo: check the Hyper threading
  private var physicalCoreNumber = System.getProperty("bigdl.engine.coreNumber",
    (Runtime.getRuntime().availableProcessors() / 2).toString()).toInt

  def coreNumber(): Int = physicalCoreNumber

  def setCoreNumber(n : Int): Unit = {
    require(n > 0)
    physicalCoreNumber = n
  }

  private var nodeNum : Option[Int] = None

  def nodeNumber(): Option[Int] = nodeNum

  def setNodeNumber(n : Int) : Unit = {
    require(n > 0)
    nodeNum = Some(n)
  }

  // =========== below is old code, will be removed after refactor===================
  /**
   * Work load parallelism
   */
  private var poolSize: Int = if(getEngineType() == MklBlas) {
    1
  } else {
    System.getProperty("dl.engine.cores",
      (Runtime.getRuntime().availableProcessors() / 2).toString()).toInt
  }


  def setCoreNum(size: Int): Unit = {
    require(size > 0)
    if (size != poolSize) {
      poolSize = size
      initEngine()
    }
  }

  def coresNum(): Int = poolSize

  private var engine: ExecutionContext = null
  /**
   * Get the ExecutionContext
   *
   * @return
   */
  def getInstance(): ExecutionContext = {
    if (engine == null) {
      initEngine()
    }
    engine
  }

  def releaseInstance[T](results : Array[Future[T]]): Seq[T] = {
    results.map(Await.result(_, Duration.Inf))
  }

  private val singleThreadEngine = new ExecutionContext {
    def execute(runnable: Runnable) {
      runnable.run()
    }

    def reportFailure(t: Throwable) {}
  }

  private def initEngine(): Unit = {
    engine = if (coresNum == 1) {
      singleThreadEngine
    } else {
      val context = new ExecutionContext {
        val threadPool = Executors.newFixedThreadPool(coresNum, new ThreadFactory {
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
      if (MKL.isMKLLoaded) {
        val results = new Array[Future[Unit]](coresNum)
        for (i <- 0 until coresNum) {
          results(i) = Future {
            MKL.setNumThreads(1)
            val tid = Thread.currentThread().getId()
            logger.info(s"Set mkl threads to 1 on thread $tid")
          }(context)
        }
        for (i <- 0 until coresNum) {
          Await.result(results(i), Duration.Inf)
        }
      }
      context
    }
  }
}
