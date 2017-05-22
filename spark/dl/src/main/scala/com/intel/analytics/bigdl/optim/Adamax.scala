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

import scala.math._
import scala.reflect.ClassTag

/**
 * An implementation of Adamax http://arxiv.org/pdf/1412.6980.pdf
 * @param learningRate learning rate
 * @param beta1 first moment coefficient
 * @param beta2 second moment coefficient
 * @param Epsilon for numerical stability
 * @tparam T
 */
class Adamax[@specialized(Float, Double) T: ClassTag](
   var learningRate: Double = 0.002,
   var beta1: Double = 0.9,
   var beta2: Double = 0.999,
   var Epsilon: Double = 1e-38
 )(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {

  /**
   * An implementation of Adamax http://arxiv.org/pdf/1412.6980.pdf
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   * @return the new x vector and the function list {fx}, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]),
               parameter: Tensor[T]): (Tensor[T], Array[T]) = {

    val lr = this.learningRate
    val beta1 = this.beta1      // Exponential decay rates 1
    val beta2 = this.beta2    // Exponential decay rates 2
    val eps = this.Epsilon

    val (fx, dfdx) = feval(parameter)

    var timestep = state.getOrElse[Int]("evalCounter", 0)

    val (_m, _u, _left, _right) =
      if (state.get[Tensor[T]]("m").isDefined) {
        (state.get[Tensor[T]]("m").get, state.get[Tensor[T]]("u").get,
          Tensor[T]().resizeAs(dfdx).zero(), Tensor[T]().resizeAs(dfdx).zero())
      } else {
        (Tensor[T]().resizeAs(dfdx).zero(), Tensor[T]().resizeAs(dfdx).zero(),
          Tensor[T]().resizeAs(dfdx).zero(), Tensor[T]().resizeAs(dfdx).zero())
      }

    timestep = timestep + 1

    _m.mul(ev.fromType[Double](beta1)).add(ev.fromType[Double](1-beta1), dfdx)
    _left.resizeAs(_u).copy(_u).mul(ev.fromType[Double](beta2))
    _right.copy(dfdx).abs().add(ev.fromType[Double](eps))
    _u.cmax(_left, _right)
    val biasCorrection1 = 1 - pow(beta1, timestep)
    val stepSize = lr / biasCorrection1
    parameter.addcdiv(ev.fromType[Double](-stepSize), _m, _u)

    state("evalCounter") = timestep
    state("m") = _m
    state("u") = _u

    (parameter, Array(fx))
  }

  override def loadFromTable(config: Table): this.type = {
    this.learningRate = config.get[Double]("learningRate").getOrElse(this.learningRate)
    this.beta1 = config.get[Double]("beta1").getOrElse(this.beta1)
    this.beta2 = config.get[Double]("beta2").getOrElse(this.beta2)
    this.Epsilon = config.get[Double]("Epsilon").getOrElse(this.Epsilon)
    this
  }

  override def clearHistory(): Unit = {
    state.delete("m")
    state.delete("u")
  }

  override def getLearningRate(): Double = this.learningRate
}
