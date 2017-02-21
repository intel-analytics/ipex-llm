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
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class SGD[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T])
  extends OptimMethod[T] {

  import SGD._

  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]), x: Tensor[T],
    config: Table, state: Table = null): (Tensor[T], Array[T]) = {

    val _state = if (state == null) config else state
    val scheduler = config.getOrElse[HyperParameterScheduler[T]]("hyperParameterScheduler",
      Default())
    val currentHyperParameter = scheduler.getAndUpdateHyperParameter(config, _state)

    val clr = ev.fromType(currentHyperParameter[Double]("learningRate"))
    val wd = currentHyperParameter[Double]("weightDecay")
    val mom = currentHyperParameter[Double]("momentum")
    val damp = currentHyperParameter[Double]("dampening")
    val nesterov = currentHyperParameter[Boolean]("nesterov")
    val lrs = currentHyperParameter[Tensor[T]]("learningRates")
    val wds = currentHyperParameter[Tensor[T]]("weightDecays")

    require(!nesterov || (mom > 0 && damp == 0),
      "Nesterov momentum requires a momentum and zero dampening")

    var (fx, dfdx) = feval(x)

    if (wd != 0) {
      dfdx.add(ev.fromType[Double](wd), x)
    } else if (wds.nElement() > 0) {
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

      if (nesterov) {
        dfdx.add(ev.fromType[Double](mom), stateDFDX)
      } else {
        dfdx = stateDFDX
      }
    }

    if (lrs.nElement() > 0) {
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
   * Abstract class for user defined hyper parameter scheduler.
   * Hyper parameters should be defined in config Table, supported parameters are:
   * learningRate, default is 1e-3
   * weightDecay, default is 0.0
   * momentum, default is 0.0
   * dampening, default is momentum
   * nesterov, default is false
   * learningRates, default is empty tensor. If defined, size must be the same as gradient Tensor.
   * weightDecays, default is empty tensor. If defined, size must be the same as gradient Tensor.
   *
   * Subclass should override updateHyperParameter(), and operate the hyper parameters.
   *
   * Notice: the clr, wd, mom and damp should be double value.
   */
  abstract class HyperParameterScheduler[T: ClassTag] {
    // current hyper parameter
    val chp = T()
    chp("learningRates") = Tensor[T]()
    chp("weightDecays") = Tensor[T]()

    private[bigdl] final def getAndUpdateHyperParameter(config: Table, state: Table): Table = {
      // init chp, copy from config or set to default value
      chp("learningRate") = config.getOrElse[Double]("learningRate", 1e-3)
      chp("weightDecay") = config.getOrElse[Double]("weightDecay", 0.0)
      chp("momentum") = config.getOrElse[Double]("momentum", 0.0)
      chp("dampening") = config.getOrElse[Double]("dampening", chp[Double]("momentum"))
      chp("nesterov") = config.getOrElse[Boolean]("nesterov", false)
      if (null != config.getOrElse[Tensor[T]]("learningRates", null)) {
        chp[Tensor[T]]("learningRates").resizeAs(config[Tensor[T]]("learningRates"))
         .copy(config[Tensor[T]]("learningRates"))
      } else {
        chp[Tensor[T]]("learningRates").set()
      }
      if (null != config.getOrElse[Tensor[T]]("weightDecays", null)) {
        chp[Tensor[T]]("weightDecays").resizeAs(config[Tensor[T]]("weightDecays"))
          .copy(config[Tensor[T]]("weightDecays"))
      } else {
        chp[Tensor[T]]("weightDecays").set()
      }
      updateHyperParameter(chp, state)
      chp
    }

    /**
     * override this method, and operate the hyperParameter.
     * Notice: hyperParameter is a hard copy of the config table, changes on hyperParameter
     * table won't change the parameter in config table.
     *
     */
    def updateHyperParameter(hyperParameter : Table, state : Table) : Unit
  }

  case class EpochSchedule[T: ClassTag](
      regimes : Array[Regime]) extends HyperParameterScheduler[T] {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val epoch = config[Int]("epoch")
      for (r <- regimes) {
        if (epoch >= r.startEpoch && epoch <= r.endEpoch) {
          config.add(r.config)
        }
      }
    }
  }
  case class Poly[T: ClassTag](
      power : Double,
      maxIteration : Int) extends HyperParameterScheduler[T] {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config[Double]("learningRate")
      val nevals = state.get[Int]("evalCounter").getOrElse(0)
      val clr = if (nevals > maxIteration) {
        0.0
      } else {
        lr * math.pow(1.0 - nevals.toDouble / maxIteration, power)
      }
      state("evalCounter") = nevals + 1
      config("learningRate") = clr
    }
  }

  case class Step[T: ClassTag](
      stepSize : Int,
      gamma : Double) extends HyperParameterScheduler[T] {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config[Double]("learningRate")
      var clr = lr
      val nevals = state.get[Int]("evalCounter").getOrElse(0)
      var i = 0
      while(i < nevals / stepSize) {
        clr *= gamma
        i += 1
      }
      state("evalCounter") = nevals + 1
      config("learningRate") = clr
    }
  }

  case class EpochDecay[T: ClassTag](
      decayType: (Int) => Double) extends HyperParameterScheduler[T] {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config[Double]("learningRate")
      var clr = lr
      val epoch = config[Int]("epoch")
      val decay = decayType(epoch)
      clr = clr * math.pow(0.1, decay)
      config("learningRate") = clr
    }
  }

  case class EpochStep[T: ClassTag](
      stepSize : Int,
      gamma : Double) extends HyperParameterScheduler[T] {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config[Double]("learningRate")
      var clr = lr
      val epoch = config[Int]("epoch")
      var i = 0
      while(i < epoch / stepSize) {
        clr *= gamma
        i += 1
      }
      config("learningRate") = clr
    }
  }

  case class Default[T: ClassTag]() extends HyperParameterScheduler[T] {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config[Double]("learningRate")
      val lrd = config.get[Double]("learningRateDecay").getOrElse(0.0)
      val nevals = state.get[Int]("evalCounter").getOrElse(0)
      config("learningRate") = lr / (1 + nevals * lrd)
      state("evalCounter") = nevals + 1
    }
  }

  case class Regime(startEpoch: Int, endEpoch: Int, config: Table)
}
