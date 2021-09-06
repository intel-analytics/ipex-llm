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
package com.intel.analytics.zoo.tfpark

import com.intel.analytics.bigdl.optim.OptimMethod
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.pipeline.api.keras.models.InternalOptimizerUtil

import scala.reflect.ClassTag

class GanOptimMethod[@specialized(Float, Double) T: ClassTag](
          val dOptim: OptimMethod[T],
          val gOptim: OptimMethod[T],
          val dSteps: Int,
          val gSteps: Int,
          gParamSize: Int)(implicit ev: TensorNumeric[T])
  extends OptimMethod[T] {

  private def isInDState(nevals: Int): Boolean = {
    nevals % (dSteps + gSteps) < dSteps
  }

  override def optimize(feval: (Tensor[T]) =>
    (T, Tensor[T]), parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    // todo try to determine state from parameter Tensor, maybe
    // define the last float to be the counter, and do not update it.
    val (fx, dfdx) = feval(parameter)
    val state = InternalOptimizerUtil.getStateFromOptiMethod(this)
    val nevals = state.getOrElse[Int]("evalCounter", 0)
    if (isInDState(nevals)) {
      dOptim.optimize(
        (_) => (fx, dfdx.narrow(1, gParamSize, parameter.nElement() - gParamSize)),
        parameter.narrow(1, gParamSize, parameter.nElement() - gParamSize))
    } else {
      gOptim.optimize((_) => (fx, dfdx.narrow(1, 1, gParamSize)),
        parameter.narrow(1, 1, gParamSize))
    }
    state("evalCounter") = nevals + 1
    (parameter, Array(fx))
  }

  override def clearHistory(): Unit = {
    dOptim.clearHistory()
    gOptim.clearHistory()
  }

  override def getLearningRate(): Double = dOptim.getLearningRate()

  override def loadFromTable(config: Table): this.type = {
    this
  }

  override def updateHyperParameter(): Unit = {
    val state = InternalOptimizerUtil.getStateFromOptiMethod(this)
    val nevals = state.getOrElse[Int]("evalCounter", 0)
    if (isInDState(nevals)) {
      dOptim.updateHyperParameter()
    } else {
      gOptim.updateHyperParameter()
    }
  }
}
