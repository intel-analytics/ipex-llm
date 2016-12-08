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
import org.apache.spark.Logging

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

  private var engine: ExecutionContext = null

  private val singletonCounter : AtomicInteger = new AtomicInteger(0)

  private var doCheckSingleton = true

  def disableCheckSingleton() : Unit = doCheckSingleton = false

  def enableCheckSingleto() : Unit = doCheckSingleton = true

  def checkSingleton() : Boolean = {
    if(!doCheckSingleton) return true

    val count = singletonCounter.incrementAndGet()
    (count == 1)
  }

  private val maxPoolSize: Int =
    System.getProperty("com.intel.analytics.bigdl.utils.Engine.maxPoolSize", "140").toInt

  private val queuePointer = new AtomicInteger(0)
  private val maxQueueSize: Int =
    System.getProperty("com.intel.analytics.bigdl.utils.Engine.maxQueueSize", "500").toInt

  private val waitQueue : Array[Future[_]] = new Array[Future[_]](maxQueueSize)

  private val threadPool : ThreadLocal[ExecutionContext] = new ThreadLocal[ExecutionContext]

  private def spawnThreadPool() : ExecutionContext = {
    new ExecutionContext {
      val pool = Executors.newFixedThreadPool(maxPoolSize, new ThreadFactory {
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

  private def getThreadPool(): ExecutionContext = {
    var pool = threadPool.get()
    if(pool == null) {
      pool = spawnThreadPool()
      threadPool.set(pool)
    }
    pool
  }

  if(Engine.engineType == MklBlas) {
    require(System.getenv("OMP_NUM_THREADS") == "1", "Under MKL_BLAS mode. Please set env " +
      "variable OMP_NUM_THREADS to 1")
    require(System.getenv("OMP_WAIT_POLICY") == "passive", "Under MKL_BLAS mode. Please set " +
      "env variable OMP_WAIT_POLICY to passive")
    require(System.getenv("KMP_BLOCKTIME") == "0", "Under MKL_BLAS mode. Please set " +
      "env variable KMP_BLOCKTIME to 0")
  } else if(Engine.engineType == MklDnn){
    require(System.getenv("OMP_NUM_THREADS") == "", "Under MKL_DNN mode. Please unset env " +
      "variable OMP_NUM_THREADS")
    require(System.getenv("OMP_WAIT_POLICY") == "", "Under MKL_DNN mode. Please unset " +
      "env variable OMP_WAIT_POLICY")
    require(System.getenv("KMP_BLOCKTIME") == "", "Under MKL_DNN mode. Please unset " +
      "env variable KMP_BLOCKTIME")
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

  /**
   * Default engine is MklBlas
   */
  private var engineType: EngineType = {
    val dlEngineType = System.getProperty("DL_ENGINE_TYPE", "MklBlas")
    if (dlEngineType.toLowerCase == "mklblas") {
      MklBlas
    } else if (dlEngineType.toLowerCase == "mkldnn") {
      MklDnn
    } else {
      throw new Error(s"Unkown DL_ENGINE_TYPE = $dlEngineType, Please use MklBlas or MklDnn")
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

  // =========== below is old code, will be removed after refactor===================
  /**
   * Work load parallelism
   */
  private var poolSize: Int = System.getProperty("dl.engine.cores",
    (Runtime.getRuntime().availableProcessors() / 2).toString()).toInt


  def setCoreNum(size: Int): Unit = {
    require(size > 0)
    if (size != poolSize) {
      poolSize = size
      initEngine()
    }
  }

  def coresNum(): Int = poolSize

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
