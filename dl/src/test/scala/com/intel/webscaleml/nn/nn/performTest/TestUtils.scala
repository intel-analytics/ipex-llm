package com.intel.webscaleml.nn.nn.performTest

import java.io._
import java.util.Properties

import com.intel.webscaleml.nn.nn.{Criterion, Module}
import com.intel.webscaleml.nn.tensor.Tensor
import org.dmg.pmml.False

import scala.reflect.runtime.universe._

/**
  * Created by yao on 6/6/16.
  */
object TestUtils {
  val iter = 30

  def isRun (): Boolean = {
    return System.getProperty("run_perform", "false").toBoolean
  }

  def testCriterionForwardPerform[T: TypeTag](input: Tensor[T], target: Tensor[T],
                                              criterion: Criterion[T], forwardIters: Int = iter): Double ={
    return testCriterionPerform(input, target, criterion, true, false, forwardIters)
  }

  def testCriterionBackwardPerform[T: TypeTag](input: Tensor[T], target: Tensor[T],
                                               criterion: Criterion[T], backwardIters: Int = iter): Double ={
    return testCriterionPerform(input, target, criterion, false, false, backwardIters)
  }

  def testModuleForwardPerform[T: TypeTag](input: Tensor[T], module: Module[T], forwardIters: Int = iter): Double ={
    return testModulePerform(input, null, module, true, false, forwardIters)
  }

  def testModuleBackwardPerform[T: TypeTag](input: Tensor[T], grad: Tensor[T],
                                               module: Module[T], backwardIters: Int = iter): Double ={
    return testModulePerform(input, grad, module, false, false, backwardIters)
  }

  def testModuleNNPackForwardPerform[T: TypeTag](input: Tensor[T], module: Module[T], forwardIters: Int = iter): Double ={
    return testModulePerform(input, null, module, true, true, forwardIters)
  }

  def testModuleNNPackBackwardPerform[T: TypeTag](input: Tensor[T], grad: Tensor[T],
                                            module: Module[T], backwardIters: Int = iter): Double ={
    return testModulePerform(input, grad, module, false, true, backwardIters)
  }

  def testModulePerform[T: TypeTag](input: Tensor[T], target: Tensor[T],
                                       module: Module[T], isForward: Boolean, isNNpack: Boolean,
                                    iters: Int = iter): Double ={
    require(typeOf[T] =:= typeOf[Double] || typeOf[T] =:= typeOf[Float]
      , "Input type can only be Tensor[Double] or Tensor[Float]")

    val filename = if (!isNNpack) "run_time.csv" else "run_time_nnpack.csv"
    val writer = new BufferedWriter(new FileWriter(new File(filename), true))

    //warm up
    val warmIter = System.getProperty("Performance.WarmIteration", "10").toInt
    for (j <- 0 until warmIter){
      if (isForward)
        module.forward(input)
      else
        module.backward(input, target)
    }

    //Calculate our module execution time
    val start = System.nanoTime()

    for (j <- 0 until iters) {
      if (isForward)
        module.forward(input)
      else
        module.backward(input, target)
    }

    val end = System.nanoTime()
    val timeMillis = (end - start) /1e6/iters

    var className = module.getClass.getSimpleName
    className = className.substring(0, className.indexOf("$"))
    writer.write(s"$className${if (isNNpack) " NNPACK"else ""}_${if (isForward) "forward" else "backward"}:")
    writer.write(f"$timeMillis%1.3f\n");
    writer.close()

    return timeMillis
  }

  def testCriterionPerform[T: TypeTag](input: Tensor[T], target: Tensor[T],
                                       module: Criterion[T], isForward: Boolean, isNNpack: Boolean,
                                       iters: Int = iter): Double ={
    require(typeOf[T] =:= typeOf[Double] || typeOf[T] =:= typeOf[Float]
      , "Input type can only be Tensor[Double] or Tensor[Float]")

    val filename = if (!isNNpack) "run_time.csv" else "run_time_nnpack.csv"
    val writer = new BufferedWriter(new FileWriter(new File(filename), true))

    //warm up
    val warmIter = System.getProperty("Performance.WarmIteration", "10").toInt
    for (j <- 0 until warmIter){
      if (isForward)
        module.forward(input, target)
      else
        module.backward(input, target)
    }

    //Calculate our module execution time
    val start = System.nanoTime()
    for (j <- 0 until iters) {
      if (isForward)
        module.forward(input, target)
      else
        module.backward(input, target)
    }
    val end = System.nanoTime()
    val timeMillis = (end - start)/1e6/iters

    var className = module.getClass.getSimpleName
    className = className.substring(0, className.indexOf("$"))
    writer.write(s"$className${if (isNNpack) "NNPACK" else ""}_${if (isForward) "forward" else "backward"}:")
    writer.write(f"$timeMillis%1.3f\n");
    writer.close()

    return timeMillis
  }
}
