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

class RMSprop[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T])
  extends OptimMethod[T] {

  /**
   * An implementation of RMSprop
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   * @param config    a table with configuration parameters for the optimizer
   *                  config("learningRate") : learning rate
   *                  config("learningRateDecay") : learning rate decay
   *                  config("decayRate"): decayRate, also called rho
   * @param state     a table describing the state of the optimizer; after each call the state
   *                  is modified
   *                  state("sumSquare"): leaky sum of squares of parameter gradients
   *                  state("rms"): and the root mean square
   * @return the new x vector and the function list, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]),
               parameter: Tensor[T], config: Table, state: Table): (Tensor[T], Array[T]) = {

    val _config = if (config == null) T() else config
    val _state = if (state == null) _config else state

    val lr = _config.getOrElse[Double]("learningRate", 1e-2)
    val lrd = _config.getOrElse[Double]("learningRateDecay", 0.0)
    val dr = _config.getOrElse[Double]("decayRate", 0.99)
    val eps = _config.getOrElse[Double]("Epsilon", 1e-8)
    val nevals = _state.getOrElse[Int]("evalCounter", 0)

    val (fx, dfdx) = feval(parameter)

    val clr = lr / (1 + nevals * lrd)

    val (_sumofsquare, _rms) =
      if (_state.get[Tensor[T]]("sumSquare").isDefined) {
        (_state.get[Tensor[T]]("sumSquare").get, _state.get[Tensor[T]]("rms").get)
      } else {
        (Tensor[T]().resizeAs(dfdx).zero(), Tensor[T]().resizeAs(dfdx).zero())
      }

    _sumofsquare.mul(ev.fromType[Double](dr)).addcmul(ev.fromType[Double](1-dr), dfdx, dfdx)
    _rms.resizeAs(_sumofsquare).copy(_sumofsquare).sqrt().add(ev.fromType[Double](eps))
    parameter.addcdiv(ev.fromType[Double](-clr), dfdx, _rms)
    _state("evalCounter") = nevals + 1
    _state("sumSquare") = _sumofsquare
    _state("rms") = _rms

    (parameter, Array(fx))
  }

  override def clearHistory(state: Table): Table = {
    state.delete("sumSquare")
    state.delete("rms")
  }
}
