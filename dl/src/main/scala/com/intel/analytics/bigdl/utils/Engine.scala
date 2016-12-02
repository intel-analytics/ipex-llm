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

import java.util.concurrent.Executors

import com.intel.analytics.bigdl.mkl.MKL
import org.apache.log4j.Logger
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}

sealed trait EngineType

case object MklBlas extends EngineType

case object MklDnn extends EngineType


/**
 * Provide appropriated thread pool based on user provided parallelism
 */
object Engine{
  /**
   * Work load parallelism
   */
  private var poolSize: Int = System.getProperty("dl.engine.cores",
    (Runtime.getRuntime().availableProcessors() / 2).toString()).toInt
  private val logger = Logger.getLogger(getClass);

  private var engine: ExecutionContext = null

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
        val threadPool = Executors.newFixedThreadPool(coresNum)

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
