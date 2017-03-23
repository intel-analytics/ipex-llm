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

class Adamax[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T])
  extends OptimMethod[T] {

  /**
   * An implementation of Adamax http://arxiv.org/pdf/1412.6980.pdf
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   * @param config    a table with hyper-parameters for the optimizer
   *                  config("learningRate") : learning rate
   *                  config("beta1") : first moment coefficient
   *                  config("beta2") : second moment coefficient
   *                  config("Epsilon"): for numerical stability
   * @param state     a table describing the state of the optimizer; after each call the state
   *                  is modified
   *                  state("m") :
   *                  state("u"):
   *                  state("denom"): A tmp tensor to hold the sqrt(v) + epsilon
   * @return the new x vector and the function list {fx}, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]),
               parameter: Tensor[T], config: Table, state: Table): (Tensor[T], Array[T]) = {

    val _config = if (config == null) T() else config
    val _state = if (state == null) _config else state

    val lr = _config.getOrElse[Double]("learningRate", 0.002)
    val beta1 = _config.getOrElse[Double]("beta1", 0.9)      // Exponential decay rates 1
    val beta2 = _config.getOrElse[Double]("beta2", 0.999)    // Exponential decay rates 2
    val eps = _config.getOrElse[Double]("Epsilon", 1e-38)

    val (fx, dfdx) = feval(parameter)

    var timestep = _state.getOrElse[Int]("evalCounter", 0)

    val (_m, _u, _left, _right) =
      if (_state.get[Tensor[T]]("m").isDefined) {
        (_state.get[Tensor[T]]("m").get, _state.get[Tensor[T]]("u").get,
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

    _state("evalCounter") = timestep
    _state("m") = _m
    _state("u") = _u

    (parameter, Array(fx))
  }

  override def clearHistory(state: Table): Table = {
    state.delete("m")
    state.delete("u")
  }
}
