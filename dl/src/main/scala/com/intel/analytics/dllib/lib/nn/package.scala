package com.intel.analytics.dllib.lib

import java.util.concurrent.Executors

import com.intel.analytics.dllib.mkl.MKL

import scala.concurrent.{Future, ExecutionContext}

package object nn {
  val coresNum : Int = System.getProperty("scala.concurrent.context.maxThreads",
    (Runtime.getRuntime().availableProcessors() / 2).toString()).toInt

  implicit val engine = if(coresNum == 1) {
    new ExecutionContext {
      def execute(runnable: Runnable) {
        runnable.run()
      }

      def reportFailure(t: Throwable) {}
    }
  } else {
    new ExecutionContext {
      val threadPool = Executors.newFixedThreadPool(coresNum)

      def execute(runnable: Runnable) {
        threadPool.submit(runnable)
      }

      def reportFailure(t: Throwable) {}
    }
  }

  if(coresNum != 1) {
    if (MKL.isMKLLoaded) {
      for (i <- 1 to coresNum) {
        Future {
          MKL.setNumThreads(1)
        }
      }
    }
  }
}
