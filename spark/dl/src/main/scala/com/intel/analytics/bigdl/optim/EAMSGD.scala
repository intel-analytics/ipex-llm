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

import com.intel.analytics.bigdl.optim.DistriOptimizer.Cache
import com.intel.analytics.bigdl.optim.SGD.{Default, LearningRateSchedule}
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
   *                config("weightDecay"): weight decay
   *                config("momentum"): momentum
   *                config("movingRate"): moving rate \alpha
   *                config("commPeriod"): sync update (communication period) \atu
   * @param state   a table describing the state of the optimizer; after each call the state
   *                is modified
   *                state("vt"): vector of movement at time t
   * @return the new x 1D vector and the function list, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]), x: Tensor[T],
    config: Table, state: Table = null, localCache: Cache[T] = null): (Tensor[T], Array[T]) = {

    val _config = if (config == null) T() else config
    val _state = if (state == null) _config else state
    val lrSchedule = config.get[LearningRateSchedule]("learningRateSchedule").getOrElse(Default())
    lrSchedule.updateHyperParameter(_config, _state)

    val mom = _config.getOrElse[Double]("momentum", 0.0)
    val wd = ev.fromType(_config.getOrElse[Double]("weightDecay", 0.0))
    val nevals = _state.get[Int]("evalCounter").getOrElse(0)
    val mr = ev.fromType(_config.getOrElse[Double]("movingRate", 0.0))
    val su = _config.getOrElse[Int]("commPeriod", 1) // sync update

    val lg = localCache.gradient                    // local gradient
    val lw = localCache.modelWeights.head           // local weight
    val led = localCache.elasticDifference          // local elastic difference

    val (fx, ged) = feval(x)                        // global elastic difference

    if (nevals % su == 0) {
      lw.add(mr, led) // TODO: do this on multi-core in parallel
      x.add(mr, ged)
    }
    if (wd != 0) lg.add(wd, lw) // TODO: do this on multi-core in parallel

    if (mom > 0) {
      if (_state.get[Tensor[T]]("vt").isDefined) {
        _state.get[Tensor[T]]("vt").get.mul(ev.fromType(mom)) // vt = mom * vt
        lw.add(_state.get[Tensor[T]]("vt").get)               // xi = xi + mom*vt
      } else {
        _state("vt") = Tensor[T]().resizeAs(lw).zero()
      }
    }

    val clr = ev.fromType(_config[Double]("clr"))
    lw.add(clr, lg) // TODO: do this in parallel

    if (mom > 0) _state.get[Tensor[T]]("vt").get.add(clr, lg)
    (x, Array(fx))
  }

  override def clearHistory(state: Table): Table = {
    state.delete("vt")
  }

  override def getHyperParameter(config: Table): String = {
    val clr = -config[Double]("clr")
    val wd = config.getOrElse[Double]("weightDecay", 0.0)
    val mom = config.getOrElse[Double]("momentum", 0.0)
    val mr = config.getOrElse[Double]("movingRate", 0.0)
    val su = config.getOrElse[Int]("commPeriod", 1)

    s"Current learning rate is $clr. " +
      {if (wd != 0) s"Current weight decay is $wd. " else ""} +
      {if (mom != 0) s"Current momentum is $mom. " else ""} +
      {if (mr != 0) s"Current moving rate (alpha) is $mr. " else ""} +
      {if (su != 0) s"Current communication period is $su. " else ""}
  }

  override def updateHyperParameter(config: Table, state: Table): Unit = {
    val lrSchedule = config.get[LearningRateSchedule]("learningRateSchedule").getOrElse(Default())
    lrSchedule.updateHyperParameter(config, state)
  }
}
