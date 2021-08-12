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

import com.intel.analytics.bigdl.mkl.hardware.Affinity
import com.intel.analytics.bigdl.mkl.{MKL, MklDnn => BackendMklDnn}
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.log4j.Logger

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}

/**
 * A thread pool wrapper, provide some helper functions for multi-threading
 *
 * TODO `TreadPool` will give 2-version of thread pool, one uses scala version (`invokeAndWait`),
 * another is provided to Java (`invokeAndWait2`). The design is weird. We should refactor this
 * class later.
 */
class ThreadPool(private var poolSize: Int) {

  import ThreadPool._


  private var mklPoolSize : Option[Int] = None
  private var threadPool: ExecutorService = null

  private var context = spawnThreadPool(poolSize)

  private def spawnThreadPool(poolSize: Int): ExecutionContext = {
    if (poolSize == 1) {
      threadPool = Executors.newFixedThreadPool(poolSize, new ThreadFactory {
        override def newThread(r: Runnable): Thread = {
          val t = Executors.defaultThreadFactory().newThread(r)
          t.setName("single-thread-computing")
          t.setDaemon(true)
          t
        }
      })

      singleThreadPool
    } else {
      new ExecutionContext {
        if (threadPool != null) threadPool.shutdown()
        threadPool = Executors.newFixedThreadPool(poolSize, new ThreadFactory {
          override def newThread(r: Runnable): Thread = {
            val t = Executors.defaultThreadFactory().newThread(r)
            t.setName("default-thread-computing " + t.getId)
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
  def setMKLThread(size: Int): this.type = this.synchronized {
    require(MKL.isMKLLoaded)
    mklPoolSize = Some(size)
    (1 to poolSize).map(i => Future {
      MKL.setNumThreads(size)
      val tid = Thread.currentThread().getId()
      logger.info(s"Set mkl threads to $size on thread $tid")
    }(context)).foreach(Await.result(_, Duration.Inf))
    this
  }

  def setMKLThreadOfMklDnnBackend(size: Int): this.type = this.synchronized {
    mklPoolSize = Some(size)


    this.invokeAndWait2((0 until 1).map(_ => () => {
      if (System.getProperty("bigdl.flushDenormalState", "true").toBoolean) {
        BackendMklDnn.setFlushDenormalState()
      }

      require(MKL.isMKLLoaded)
      require(BackendMklDnn.isLoaded)

      MKL.setNumThreads(size)
      BackendMklDnn.setNumThreads(size)
      if (!System.getProperty("bigdl.disableOmpAffinity", "false").toBoolean) {
        Affinity.setOmpAffinity()
      }
    }))

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

  private type JavaFuture[T] = java.util.concurrent.Future[T]

  /**
   * Use java future to execute the tasks. It will be blocking until tasks completed.
   * If any task throws an exception, it will throw that exception in caller.
   *
   * @param tasks task sequence. each task's return type is T
   * @param timeout the maximum time to wait
   * @param timeUnit the time unit for the timeout
   * @tparam T return type of tasks
   * @return a sequence of Futures representing the tasks.
   */
  def invokeAndWait2[T](tasks: Seq[() => T], timeout: Long = Long.MaxValue,
    timeUnit: TimeUnit = TimeUnit.NANOSECONDS): mutable.Buffer[JavaFuture[T]] = {
    val callables = tasks.map(task => new Callable[T] {
      override def call(): T = {
        task()
      }
    })

    val resultFutures = threadPool.invokeAll(callables.asJava, timeout, timeUnit)

    // we should check all the future in the list, if any task has an exception,
    // we should throw it.
    var i = 0
    while (i < resultFutures.size()) {
      try {
        resultFutures.get(i).get()
      } catch {
        case t: ExecutionException => throw t.getCause
        case i: InterruptedException => throw i.getCause
      }
      i += 1
    }

    resultFutures.asScala
  }

  def invoke2[T](tasks: Seq[() => T]): Seq[JavaFuture[T]] = {
    tasks.map(task => new Callable[T] {
      override def call(): T = {
        try {
          task()
        } catch {
          case t : Throwable =>
            logger.error("Error: " + ExceptionUtils.getStackTrace(t))
            throw t
        }
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
      try {
        task()
      } catch {
        case t : Throwable =>
          logger.error("Error: " + ExceptionUtils.getStackTrace(t))
          throw t
      }
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
  def setPoolSize(size: Int): this.type = this.synchronized {
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

