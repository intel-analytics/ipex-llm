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
 * Adadelta implementation for SGD: http://arxiv.org/abs/1212.5701
 * @param decayRate decayRate, also called interpolation parameter rho
 * @param Epsilon for numerical stability
 * @tparam T
 */
class Adadelta[@specialized(Float, Double) T: ClassTag](
   var decayRate: Double = 0.9,
   var Epsilon: Double = 1e-10
 )(implicit ev: TensorNumeric[T])
  extends OptimMethod[T] {

  /**
   * Adadelta implementation for SGD: http://arxiv.org/abs/1212.5701
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   *                  state("paramVariance") : vector of temporal variances of parameters
   *                  state("accDelta"): vector of accumulated delta of gradients
   * @return the new x vector and the function list {fx}, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]),
               parameter: Tensor[T]): (Tensor[T], Array[T]) = {

    val nevals = state.getOrElse[Int]("evalCounter", 0)
    val dr = this.decayRate
    val eps = this.Epsilon

    val (fx, dfdx) = feval(parameter)

    val (_paramVariance, _paramStd, _delta, _accDelta) =
      if (state.get[Tensor[T]]("paramVariance").isDefined) {
        (state.get[Tensor[T]]("paramVariance").get, state.get[Tensor[T]]("paramStd").get,
          state.get[Tensor[T]]("delta").get, state.get[Tensor[T]]("accDelta").get)
      } else {
        (Tensor[T]().resizeAs(dfdx).zero(), Tensor[T]().resizeAs(dfdx).zero(),
          Tensor[T]().resizeAs(dfdx).zero(), Tensor[T]().resizeAs(dfdx).zero())
      }

    _paramVariance.mul(ev.fromType[Double](dr)).addcmul(ev.fromType[Double](1-dr), dfdx, dfdx)
    _paramStd.copy(_paramVariance).add(ev.fromType[Double](eps)).sqrt()
    _delta.copy(_accDelta).add(ev.fromType[Double](eps)).sqrt()
      .cdiv(_paramStd).cmul(dfdx)
    parameter.add(ev.fromType[Double](-1), _delta)
    _accDelta.mul(ev.fromType[Double](dr)).addcmul(ev.fromType[Double](1-dr), _delta, _delta)
    state("evalCounter") = nevals + 1
    state("paramVariance") = _paramVariance
    state("paramStd") = _paramStd
    state("delta") = _delta
    state("accDelta") = _accDelta

    (parameter, Array(fx))
  }

  override def loadFromTable(config: Table): this.type = {
    this.decayRate = config.get[Double]("decayRate").getOrElse(this.decayRate)
    this.Epsilon = config.get[Double]("Epsilon").getOrElse(this.Epsilon)
    this
  }

  override def clearHistory(): Unit = {
    state.delete("paramVariance")
    state.delete("paramStd")
    state.delete("delta")
    state.delete("accDelta")
  }

  override def getLearningRate(): Double = 0.0
}
