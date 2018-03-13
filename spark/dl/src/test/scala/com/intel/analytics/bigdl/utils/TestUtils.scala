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

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.nn.Sequential
import com.intel.analytics.bigdl.nn.keras.{InputLayer, KerasLayer, Sequential => KSequential}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.scalatest.exceptions.TestCanceledException

import scala.reflect.ClassTag

object TestUtils {

  /**
   * Compare the output of `computeOutputShape` with the `forward` result
   */
  def compareOutputShape(layer: AbstractModule[Activity, Activity, Float],
                         inputShapeWithoutBatch: Shape): Boolean = {
    val inputData = Tensor[Float](Array(2) ++ inputShapeWithoutBatch.toSingle()).randn()
    val runnableLayer = layer match {
      case k: KerasLayer[_, _, _] =>
        if (!k.isBuilt()) {
          k.build(KerasLayer.addBatch(inputShapeWithoutBatch))
        }
        k
      case a: AbstractModule[_, _, _] => a
    }
    val calcOutputShape = runnableLayer.computeOutputShape(
      KerasLayer.addBatch(inputShapeWithoutBatch)).toSingle()
    val forwardOutputShape = runnableLayer.forward(inputData).toTensor[Float].size()
    calcOutputShape.slice(1, calcOutputShape.length).sameElements(
      forwardOutputShape.slice(1, forwardOutputShape.length))
  }

  /**
   * Process different paths format under windows and linux
   *
   * @param path
   * @return
   */
  def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  /**
   * Some test case cannot run on windows, cancel such test cases
   */
  def cancelOnWindows(): Unit = {
    if (System.getProperty("os.name").toLowerCase().contains("win")) {
      throw new TestCanceledException("This case should not be run on windows", 3)
    }
  }

  /**
   * This function returns the function value, partial derivatives
   * and Hessian of the (general dimension) rosenbrock function, given by:
   * f(x) = sum_{i=1:D-1} 100*(x(i+1) - x(i)^2)^2 + (1-x(i)) ^^ 2
   * where D is the dimension of x. The true minimum is 0 at x = (1 1 ... 1).
   *
   * See more about rosenbrock function at
   * https://en.wikipedia.org/wiki/Rosenbrock_function
   *
   * @param x
   */
  def rosenBrock(x: Tensor[Double]): (Double, Tensor[Double]) = {
    // (1) compute f(x)
    val d = x.size(1)

    // x1 = x(i)
    val x1 = Tensor[Double](d - 1).copy(x.narrow(1, 1, d - 1))
    // x(i + 1) - x(i)^2
    x1.cmul(x1).mul(-1).add(x.narrow(1, 2, d - 1))
    // 100 * (x(i + 1) - x(i)^2)^2
    x1.cmul(x1).mul(100)

    // x0 = x(i)
    val x0 = Tensor[Double](d - 1).copy(x.narrow(1, 1, d - 1))
    // 1-x(i)
    x0.mul(-1).add(1)
    x0.cmul(x0)
    // 100*(x(i+1) - x(i)^2)^2 + (1-x(i))^2
    x1.add(x0)

    val fout = x1.sum()

    // (2) compute f(x)/dx
    val dxout = Tensor[Double]().resizeAs(x).zero()
    // df(1:D-1) = - 400*x(1:D-1).*(x(2:D)-x(1:D-1).^2) - 2*(1-x(1:D-1));
    x1.copy(x.narrow(1, 1, d - 1))
    x1.cmul(x1).mul(-1).add(x.narrow(1, 2, d - 1)).cmul(x.narrow(1, 1, d - 1)).mul(-400)
    x0.copy(x.narrow(1, 1, d - 1)).mul(-1).add(1).mul(-2)
    x1.add(x0)
    dxout.narrow(1, 1, d - 1).copy(x1)

    // df(2:D) = df(2:D) + 200*(x(2:D)-x(1:D-1).^2);
    x0.copy(x.narrow(1, 1, d - 1))
    x0.cmul(x0).mul(-1).add(x.narrow(1, 2, d - 1)).mul(200)
    dxout.narrow(1, 2, d - 1).add(x0)

    (fout, dxout)
  }
}

class ExceptionTest[T: ClassTag](failCountNumberLists: Array[Int], sleep: Boolean)
  (implicit ev: TensorNumeric[T])
  extends TensorModule[T]  {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    this.output = input
    if (failCountNumberLists.contains(ExceptionTest.count.incrementAndGet())) {
      if (sleep) {
        Thread.sleep(10000)
      }
      throw new Exception("Fail task")
    }
    this.output
  }
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    this.gradInput = gradOutput
    this.gradInput
  }

  override def toString(): String = {
    s"nn.ExceptionTest"
  }
}

object ExceptionTest {
  var count = new AtomicInteger(0)

  def resetCount(): Unit = {
    count.set(0)
  }
}
