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

class Adadelta[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T])
  extends OptimMethod[T] {

  /**
   * Adadelta implementation for SGD: http://arxiv.org/abs/1212.5701
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   * @param config    a table with hyper-parameters for the optimizer
   *                  config("learningRate") : learning rate
   *                  config("learningRateDecay") : learning rate decay
   *                  config("decayRate"): decayRate, also called interpolation parameter rho
   *                  config("Epsilon"): for numerical stability
   * @param state     a table describing the state of the optimizer; after each call the state
   *                  is modified
   *                  state("paramVariance") : vector of temporal variances of parameters
   *                  state("accDelta"): vector of accumulated delta of gradients
   * @return the new x vector and the function list {fx}, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]),
               parameter: Tensor[T], config: Table, state: Table): (Tensor[T], Array[T]) = {

    val _config = if (config == null) T() else config
    val _state = if (state == null) _config else state

    val nevals = _state.getOrElse[Int]("evalCounter", 0)
    val dr = _config.getOrElse[Double]("decayRate", 0.9)
    val eps = _config.getOrElse[Double]("Epsilon", 1e-10)

    val (fx, dfdx) = feval(parameter)

    val (_paramVariance, _paramStd, _delta, _accDelta) =
      if (_state.get[Tensor[T]]("paramVariance").isDefined) {
        (_state.get[Tensor[T]]("paramVariance").get, _state.get[Tensor[T]]("paramStd").get,
          _state.get[Tensor[T]]("delta").get, _state.get[Tensor[T]]("accDelta").get)
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
    _state("evalCounter") = nevals + 1
    _state("paramVariance") = _paramVariance
    _state("paramStd") = _paramStd
    _state("delta") = _delta
    _state("accDelta") = _accDelta

    (parameter, Array(fx))
  }

  override def clearHistory(state: Table): Table = {
    state.delete("paramVariance")
    state.delete("paramStd")
    state.delete("delta")
    state.delete("accDelta")
  }
}
