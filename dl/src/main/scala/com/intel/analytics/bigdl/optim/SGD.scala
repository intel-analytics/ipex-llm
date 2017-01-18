/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class SGD[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T])
  extends OptimMethod[T] {

  import SGD._

  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]), x: Tensor[T],
    config: Table, state: Table = null): (Tensor[T], Array[T]) = {

    val _state = if (state == null) config else state
    val lrSchedule = config.get[HyperParameterScheduler]("learningRateSchedule").getOrElse(Default())
    lrSchedule.updateHyperParameter(config, _state)

    val clr = ev.fromType(config.getOrElse[Double]("clr",
      config.getOrElse[Double]("learningRate", 1e-3)))
    val wd = config.getOrElse[Double]("wd", config.getOrElse[Double]("weightDecay", 0.0))
    val mom = config.getOrElse[Double]("mom", config.getOrElse[Double]("momentum", 0.0))
    val damp = config.getOrElse[Double]("damp", config.getOrElse[Double]("dampening", mom))
    val cntv = config.getOrElse[Boolean]("cntv", config.getOrElse[Boolean]("nesterov", false))
    val lrs = config.getOrElse[Tensor[T]]("lrs", config.getOrElse[Tensor[T]]("learningRates", null))
    val wds = config.getOrElse[Tensor[T]]("wds", config.getOrElse[Tensor[T]]("weightDecays", null))

    require(!cntv || (mom > 0 && damp == 0),
      "Nesterov momentum requires a momentum and zero dampening")

    var (fx, dfdx) = feval(x)

    if (wd != 0) {
      dfdx.add(ev.fromType[Double](wd), x)
    } else if (wds != null) {
      val decayParameters = _state.get[Tensor[T]]("decayParameters").getOrElse({
        val DP = Tensor[T]().resizeAs(dfdx)
        _state("decayParameters") = DP
        DP
      })
      decayParameters.copy(wds).cmul(x)
      dfdx.add(decayParameters)
    }

    if (mom != 0) {
      val stateDFDX = _state.get[Tensor[T]]("dfdx") match {
        case None =>
          val DFDX = Tensor[T]().resizeAs(dfdx).copy(dfdx)
          _state("dfdx") = DFDX
          DFDX
        case s: Some[Tensor[T]] => s.get.mul(ev.fromType[Double](mom)).
          add(ev.fromType[Double](1 - damp), dfdx)
      }

      if (cntv) {
        dfdx.add(ev.fromType[Double](mom), stateDFDX)
      } else {
        dfdx = stateDFDX
      }
    }

    if (lrs != null) {
      val deltaParameters = _state.get[Tensor[T]]("deltaParameters").getOrElse({
        val deltaP = Tensor[T]().resizeAs(dfdx)
        _state("deltaParameters") = deltaP
        deltaP
      })
      deltaParameters.copy(lrs).cmul(dfdx)
      x.add(ev.negative(clr), deltaParameters)
    } else {
      x.add(ev.negative(clr), dfdx)
    }


    (x, Array(fx))
  }

  override def clearHistory(state: Table): Table = {
    state.delete("decayParameters")
    state.delete("dfdx")
    state.delete("deltaParameters")
  }
}

object SGD {

  /**
   * A trait for user defined hyper parameter scheduler.
   * Hyper parameters should be defined in config Table, supported parameters are:
   * learningRate, default is 1e-3
   * weightDecay, default is 0.0
   * momentum, default is 0.0
   * dampening, default is momentum
   * nesterov, default is false
   * learningRates, default is null. If defined, size must be the same as gradient Tensor.
   * weightDecays, default is null. If defined, size must be the same as gradient Tensor.
   *
   * Please don't overwrite the base hyper parameters list above, and use below mapping
   * to define the current hyper parameters.
   *
   * config("clr") = current learning rate
   * config("wd") = current weight decay
   * config("mom") = current momentum
   * config("damp") = current dampening
   * config("cntv") = current nesterov
   * config("lrs") = current learning rates
   * config("wds") = current weight decays
   *
   * Notice: the clr, wd, mom and damp should be double value.
   */
  trait HyperParameterScheduler {
    def updateHyperParameter(config : Table, state : Table) : Unit
  }

  case class EpochSchedule(regimes : Array[Regime]) extends HyperParameterScheduler {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val epoch = config[Int]("epoch")
      for (r <- regimes) {
        if (epoch >= r.startEpoch && epoch <= r.endEpoch) {
          config.add(r.config)
        }
      }
      config("clr") = config.get[Double]("learningRate").getOrElse(1e-3)
    }
  }
  case class Poly(power : Double, maxIteration : Int) extends HyperParameterScheduler {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config.get[Double]("learningRate").getOrElse(1e-3)
      val nevals = state.get[Int]("evalCounter").getOrElse(0)
      val clr = if (nevals > maxIteration) {
        0.0
      } else {
        lr * math.pow(1.0 - nevals.toDouble / maxIteration, power)
      }
      state("evalCounter") = nevals + 1
      config("clr") = clr
    }
  }

  case class Step(stepSize : Int, gamma : Double) extends HyperParameterScheduler {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config.get[Double]("learningRate").getOrElse(1e-3)
      var clr = lr
      val nevals = state.get[Int]("evalCounter").getOrElse(0)
      var i = 0
      while(i < nevals / stepSize) {
        clr *= gamma
        i += 1
      }
      state("evalCounter") = nevals + 1
      config("clr") = clr
    }
  }

  case class EpochDecay(decayType: (Int) => Double) extends HyperParameterScheduler {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config.get[Double]("learningRate").getOrElse(1e-1)
      var clr = lr
      val epoch = config[Int]("epoch")
      val decay = decayType(epoch)
      clr = clr * math.pow(0.1, decay)
      config("clr") = clr
    }
  }

  case class EpochStep(stepSize : Int, gamma : Double) extends HyperParameterScheduler {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config.get[Double]("learningRate").getOrElse(1e-3)
      var clr = lr
      val epoch = config[Int]("epoch")
      var i = 0
      while(i < epoch / stepSize) {
        clr *= gamma
        i += 1
      }
      config("clr") = clr
    }
  }

  case class Default() extends HyperParameterScheduler {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config.get[Double]("learningRate").getOrElse(1e-3)
      val lrd = config.get[Double]("learningRateDecay").getOrElse(0.0)
      val nevals = state.get[Int]("evalCounter").getOrElse(0)
      config("clr") = lr / (1 + nevals * lrd)
      state("evalCounter") = nevals + 1
    }
  }

  case class Regime(startEpoch: Int, endEpoch: Int, config: Table)
}
