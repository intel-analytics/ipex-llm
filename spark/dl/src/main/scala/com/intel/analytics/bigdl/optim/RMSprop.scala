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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * An implementation of RMSprop
 * @param learningRate learning rate
 * @param learningRateDecay learning rate decay
 * @param decayRate decayRate, also called rho
 * @param Epsilon for numerical stability
 * @tparam T
 */
class RMSprop[@specialized(Float, Double) T: ClassTag](
    var learningRate: Double = 1e-2,
    var learningRateDecay: Double = 0.0,
    var decayRate: Double = 0.99,
    var Epsilon: Double = 1e-8
  )(implicit ev: TensorNumeric[T])
  extends OptimMethod[T] {

  /**
   * An implementation of RMSprop
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   * @return the new x vector and the function list, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]),
               parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    val lr = this.learningRate
    val lrd = this.learningRateDecay
    val dr = this.decayRate
    val eps = this.Epsilon
    val nevals = state.getOrElse[Int]("evalCounter", 0)

    val (fx, dfdx) = feval(parameter)

    val clr = lr / (1 + nevals * lrd)

    val (_sumofsquare, _rms) =
      if (state.get[Tensor[T]]("sumSquare").isDefined) {
        (state.get[Tensor[T]]("sumSquare").get, state.get[Tensor[T]]("rms").get)
      } else {
        (Tensor[T]().resizeAs(dfdx).zero(), Tensor[T]().resizeAs(dfdx).zero())
      }

    _sumofsquare.mul(ev.fromType[Double](dr)).addcmul(ev.fromType[Double](1-dr), dfdx, dfdx)
    _rms.sqrt(_sumofsquare).add(ev.fromType[Double](eps))
    parameter.addcdiv(ev.fromType[Double](-clr), dfdx, _rms)
    state("evalCounter") = nevals + 1
    state("sumSquare") = _sumofsquare
    state("rms") = _rms

    (parameter, Array(fx))
  }


  override def loadFromTable(config: Table): this.type = {
    this.learningRate = config.get[Double]("learningRate").getOrElse(this.learningRate)
    this.learningRateDecay = config.get[Double]("learningRateDecay")
      .getOrElse(this.learningRateDecay)
    this.decayRate = config.get[Double]("decayRate").getOrElse(this.decayRate)
    this.Epsilon = config.get[Double]("Epsilon").getOrElse(this.Epsilon)
    this
  }

  override def clearHistory(): Unit = {
    state.delete("sumSquare")
    state.delete("rms")
  }

  override def getLearningRate(): Double = this.learningRate
}
