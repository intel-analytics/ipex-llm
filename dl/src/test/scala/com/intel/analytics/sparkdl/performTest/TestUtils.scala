package com.intel.analytics.sparkdl.performTest

import java.io._

import scala.reflect.runtime.universe._

/**
  * Created by yao on 6/6/16.
  */
object TestUtils {
  val iter = 10

  def isRun (): Boolean = {
    return System.getProperty("run_perform", "false").toBoolean
  }

  def testMathOperation[T: TypeTag](doOperation: () => T, printString: String, iters: Int = iter): Double = {
    //require(typeOf[T] =:= typeOf[Double] || typeOf[T] =:= typeOf[Float]
    //  , "Input type can only be Tensor[Double] or Tensor[Float]")

    val filename = "run_time.csv"
    val writer = new BufferedWriter(new FileWriter(new File(filename), true))

    //warm up
    val warmIter = System.getProperty("Performance.WarmIteration", "5").toInt
    for (j <- 0 until warmIter){
      doOperation()
    }

    //Calculate our module execution time
    val start = System.nanoTime()
    for (j <- 0 until iters) {
      doOperation()
    }
    val end = System.nanoTime()
    val timeMillis = (end - start) /1e6/iters

    writer.write(printString)
    writer.write(f", $timeMillis%1.3f\n");
    writer.close()

    return timeMillis
  }
}
