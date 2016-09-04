package com.intel.analytics.dllib.lib.nn.performTest

/**
  * Created by yao on 6/6/16.
  */
object PerformTestData {
  val forwardIterations = System.getProperty("Performance.ForwardIteration", "30").toInt
  val backwardIterations = System.getProperty("Performance.BackwardIteration", "30").toInt
}