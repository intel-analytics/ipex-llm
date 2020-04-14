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

import java.util.concurrent.{ExecutorService, Executors, ThreadFactory}

import com.intel.analytics.zoo.core.TFNetNative
import com.intel.analytics.zoo.pipeline.api.net.NetUtils
import jep.{JepConfig, JepException, NamingConventionClassEnquirer, SharedInterpreter}
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.log4j.{Level, Logger}

import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.duration.Duration

object PythonInterpreter {
  protected val logger = Logger.getLogger(this.getClass)

  private var threadPool: ExecutorService = null

  private val context = new ExecutionContext {
    threadPool = Executors.newFixedThreadPool(1, new ThreadFactory {
      override def newThread(r: Runnable): Thread = {
        val t = Executors.defaultThreadFactory().newThread(r)
        t.setName("jep-thread " + t.getId)
        t.setDaemon(true)
        t
      }
    })

    def execute(runnable: Runnable) {
      threadPool.submit(runnable)
    }

    def reportFailure(t: Throwable): Unit = {
      throw t
    }
  }
  def getSharedInterpreter(): SharedInterpreter = {
    sharedInterpreter
  }

  private val sharedInterpreter: SharedInterpreter = createInterpreter()
  private def createInterpreter(): SharedInterpreter = {
    if (System.getenv("PYTHONHOME") == null) {
      throw new RuntimeException("PYTHONHOME is unset, please set PYTHONHOME first.")
    }
    // Load TFNet before create interpreter, or the TFNet will throw an OMP error #13
    TFNetNative.isLoaded
    val createInterp = () => {
      val config: JepConfig = new JepConfig()
        config.setClassEnquirer(new NamingConventionClassEnquirer())
        SharedInterpreter.setConfig(config)
        val sharedInterpreter = new SharedInterpreter()
        sharedInterpreter
      }
    logger.debug("Creating jep interpreter...")
    threadExecute(createInterp)
  }

  private def threadExecute[T](task: () => T,
                               timeout: Duration = Duration.Inf): T = {
    try {
      val re = Array(task).map(t => Future {
        t()
      }(context)).map(future => {
        Await.result(future, timeout)
      })
      re(0)
    } catch {
      case t: Throwable =>
        // Don't use logger here, or spark local will stuck when catch an exception.
        println("Error: " + ExceptionUtils.getStackTrace(t))
        throw new JepException(t)
    }
  }

  def exec(s: String): Unit = {
    logger.debug(s"jep exec ${s}")
    val func = () => {
      sharedInterpreter.exec(s)
    }
    threadExecute(func)
  }

  def set(s: String, o: AnyRef): Unit = {
    logger.debug(s"jep set ${s}")
    val func = () => {
      sharedInterpreter.set(s, o)
    }
    threadExecute(func)
  }

  def getValue[T](name: String): T = {
    logger.debug(s"jep getValue ${name}")
    val func = () => {
      val re = sharedInterpreter.getValue(name)
      re
    }
    threadExecute(func).asInstanceOf[T]
  }
}
