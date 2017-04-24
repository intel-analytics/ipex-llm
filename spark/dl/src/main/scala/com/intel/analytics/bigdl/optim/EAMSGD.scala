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
 * An implementation of EASGD/EAMSGD.
 * EAMSGD is based on the Nesterov's momentum scheme,
 * When mom==0, it is the EASGD.
 * See the original paper: https://cs.nyu.edu/~zsx/nips2015.pdf
 * @param ev numeric operator
 * @tparam T data type
 */
class EAMSGD[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T])
  extends OptimMethod[T] {

  /**
   * Elastic Averaging (Momentum) SGD implementation
   *
   * @param feval   a function that takes a single input (X), the point of a evaluation,
   *                and returns f(X) and df/dX
   * @param x       the initial point
   * @param config  a table with configuration parameters for the optimizer
   *                config("learningRate"): learning rate
   *                config("learningRateDecay"): learning rate decay
   *                config("learningRateDecayPower"): learning rate decay power
   *                config("weightDecay"): weight decay
   *                config("momentum"): momentum
   *                config("movingRate"): moving rate \alpha
   *                config("commPeriod"): sync update (communication period) \atu
   * @param state   a table describing the state of the optimizer; after each call the state
   *                is modified
   *                state("vt"): vector of movement at time t
   *                state("suw"): vector of weight for communication with the center variable
   *                state("sug"): vector of gradient for communication with the center variable
   * @return the new x 1D vector and the function list, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]), x: Tensor[T],
    config: Table, state: Table = null): (Tensor[T], Array[T]) = {

    val _config = if (config == null) T() else config
    val _state = if (state == null) _config else state

    val lr = _config.getOrElse[Double]("learningRate", 1e-3)
    val lrd = _config.getOrElse[Double]("learningRateDecay", 0.0)
    val lrp = _config.getOrElse[Double]("learningRateDecayPower", 1.0)
    val mom = _config.getOrElse[Double]("momentum", 0.0)
    val wd = _config.getOrElse[Double]("weightDecay", 0.0)

    val nevals = _state.get[Int]("evalCounter").getOrElse(1)

    val mr = _config.getOrElse[Double]("movingRate", 0.0)
    val su = _config.getOrElse[Int]("commPeriod", 1) // sync update

    if (nevals % su == 0) { // sync with center parameter server
      if (!_state.get[Tensor[T]]("suw").isDefined) {
        _state("suw") = Tensor[T]().resizeAs(x).zero()
        _state("sug") = Tensor[T]().resizeAs(x).zero()
      }
      // 1. receive w* from pServer
      // 2. calculate elastic difference: alpha*(w - w*) locally
      // 3. update elastic difference to local and remote
      // To cope with the current code base, 1 and 2 are carried out in DistriOptimizer
    }

    if (mom > 0) {
      if (_state.get[Tensor[T]]("vt").isDefined) {
        _state.get[Tensor[T]]("vt").get.mul(ev.fromType[Double](mom)) // vt = mom * vt
        x.add(_state.get[Tensor[T]]("vt").get) // xi = xi + mom*vt
      } else {
        _state("vt") = Tensor[T]().resizeAs(x).zero()
      }
    }
    var (fx, dfdx) = feval(x)

    if (wd != 0) {
      dfdx.add(ev.fromType[Double](wd), x)
    }
    var clr = -lr
    if (lrd != 0 && lrp > 0) {
      clr = clr / math.pow(nevals * lrd + 1, lrp)
    }
//    x.add(ev.fromType[Double](clr), dfdx)
    x.add(dfdx)
    if (mom > 0) {
      _state.get[Tensor[T]]("vt").get.add(ev.fromType[Double](clr), dfdx)
    }

    _state("evalCounter") = nevals + 1

    (x, Array(fx))
  }

  override def clearHistory(state: Table): Table = {
    state.delete("vt")
    state.delete("suw")
    state.delete("sug")
  }

  override def getHyperParameter(config: Table): String = {
    val lr = -config[Double]("learningRate")
    val lrd = config.getOrElse[Double]("learningRateDecay", 0.0)
    val lrp = config.getOrElse[Double]("learningRateDecayPower", 0.0)
    val mom = config.getOrElse[Double]("momentum", 0.0)
    val wd = config.getOrElse[Double]("weightDecay", 0.0)

    val mr = config.getOrElse[Double]("movingRate", 0.0)
    val cp = config.getOrElse[Int]("commPeriod", 1)

    s"Current learning rate is $lr. " +
      {if (wd != 0) s"Current weight decay is $wd. " else ""} +
      {if (mom != 0) s"Current momentum is $mom. " else ""} +
      {if (lrd != 0) s"Current learning rate decay is $lrd. " else ""} +
      {if (lrp != 0) s"Current learning rate decay power is $lrp. " else ""} +
      {if (mr != 0) s"Current moving rate (alpha) is $mr. " else ""} +
      {if (cp != 0) s"Current communication period is $cp. " else ""}
  }
}
