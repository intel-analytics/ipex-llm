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
    val lrSchedule = config.get[LearningRateSchedule]("learningRateSchedule").getOrElse(Default())
    lrSchedule.updateHyperParameter(config, _state)

    val wd = config.get[Double]("weightDecay").getOrElse(0.0)
    val mom = config.get[Double]("momentum").getOrElse(0.0)
    val damp = config.get[Double]("dampening").getOrElse(mom)
    val nesterov = config.get[Boolean]("nesterov").getOrElse(false)
    val lrs = config.get[Tensor[T]]("learningRates").getOrElse(null)
    val wds = config.get[Tensor[T]]("weightDecays").getOrElse(null)

    require(!nesterov || (mom > 0 && damp == 0),
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

      if (nesterov) {
        dfdx.add(ev.fromType[Double](mom), stateDFDX)
      } else {
        dfdx = stateDFDX
      }
    }

    val clr = ev.fromType(config[Double]("clr"))
    if (lrs != null) {
      val deltaParameters = _state.get[Tensor[T]]("deltaParameters").getOrElse({
        val deltaP = Tensor[T]().resizeAs(dfdx)
        _state("deltaParameters") = deltaP
        deltaP
      })
      deltaParameters.copy(lrs).cmul(dfdx)
      x.add(clr, deltaParameters)
    } else {
      x.add(clr, dfdx)
    }


    (x, Array(fx))
  }

  override def clearHistory(state: Table): Table = {
    state.delete("decayParameters")
    state.delete("dfdx")
    state.delete("deltaParameters")
  }

  /**
   * return an string of current hyperParameter.
   */
  override def getHyperParameter(config: Table): String = {
    val clr = -config[Double]("clr")
    val wd = config.get[Double]("weightDecay").getOrElse(0.0)
    val mom = config.get[Double]("momentum").getOrElse(0.0)
    val damp = config.get[Double]("dampening").getOrElse(mom)
    val nesterov = config.get[Boolean]("nesterov").getOrElse(false)
    val lrs = config.get[Tensor[T]]("learningRates").getOrElse(null)
    val wds = config.get[Tensor[T]]("weightDecays").getOrElse(null)
    s"Current learning rate is $clr. " +
      {if (wd != 0) s"Current weight decay is $wd. " else ""} +
      {if (mom != 0) s"Current momentum is $mom. " else ""} +
      {if (damp != 0) s"Current dampening is $damp. " else ""} +
      {if (nesterov) s"Current nesterov is true. " else ""} +
      {if (null != lrs) s"Current learningRates is a Tensor. " else ""} +
      {if (null != wds) s"Current weightDecays is a Tensor. " else ""}
  }

  override def updateHyperParameter(config: Table, state: Table): Unit = {
    val lrSchedule = config.get[LearningRateSchedule]("learningRateSchedule").getOrElse(Default())
    lrSchedule.updateHyperParameter(config, state)
  }
}

object SGD {
  trait LearningRateSchedule {
    def updateHyperParameter(config : Table, state : Table) : Unit
  }

  /**
   * [[EpochSchedule]] is a learning rate schedule which configure the learning
   * rate according to some pre-defined [[Regime]]. If the running epoch is within
   * the interval of a regime `r` [r.startEpoch, r.endEpoch], then the learning
   * rate will take the "learningRate" in r.config.
   *
   * @param regimes an array of pre-defined [[Regime]].
   */
  case class EpochSchedule(regimes : Array[Regime]) extends LearningRateSchedule {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val epoch = state[Int]("epoch")
      for (r <- regimes) {
        if (epoch >= r.startEpoch && epoch <= r.endEpoch) {
          config.add(r.config)
        }
      }
      config("clr") = -config.get[Double]("learningRate").getOrElse(1e-3)
    }
  }
  case class Poly(power : Double, maxIteration : Int) extends LearningRateSchedule {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config.get[Double]("learningRate").getOrElse(1e-3)
      val nevals = state.get[Int]("evalCounter").getOrElse(0)
      val clr = if (nevals > maxIteration) {
        0.0
      } else {
        -lr * math.pow(1.0 - nevals.toDouble / maxIteration, power)
      }
      println(s"iteration is : ${nevals}. current learning rate is $clr")
      state("evalCounter") = nevals + 1
      config("clr") = clr
    }
  }

  case class Step(stepSize : Int, gamma : Double) extends LearningRateSchedule {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config.get[Double]("learningRate").getOrElse(1e-3)
      var clr = -lr
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

  /**
   * It is an epoch decay learning rate schedule
   * The learning rate decays through a function argument on number of run epochs
   *
   * l_{n + 1} = l_{n} * 0.1 `^` decayType(epoch)
   *
   * @param decayType is a function with number of run epochs as the argument
   */
  case class EpochDecay(decayType: (Int) => Double) extends LearningRateSchedule {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config.get[Double]("learningRate").getOrElse(1e-1)
      var clr = -lr
      val epoch = state[Int]("epoch")
      val decay = decayType(epoch)
      clr = clr * math.pow(0.1, decay)
      config("clr") = clr
    }
  }

  /**
   * [[EpochStep]] is a learning rate schedule, which rescale the learning rate by `gamma`
   * for each `stepSize` epochs.
   *
   * @param stepSize For how many epochs to update the learning rate once
   * @param gamma the rescale factor
   */
  case class EpochStep(stepSize : Int, gamma : Double) extends LearningRateSchedule {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config.get[Double]("learningRate").getOrElse(1e-3)
      var clr = -lr
      val epoch = state[Int]("epoch")
      var i = 0
      while(i < epoch / stepSize) {
        clr *= gamma
        i += 1
      }
      config("clr") = clr
    }
  }

  /**
   * It is the default learning rate schedule.
   * For each iteration, the learning rate would
   * update with the following formula:
   *
   * l_{n + 1} = l / (1 + n * learning_rate_decay)
   *
   * where `l` is the initial learning rate
   */
  case class Default() extends LearningRateSchedule {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config.get[Double]("learningRate").getOrElse(1e-3)
      val lrd = config.get[Double]("learningRateDecay").getOrElse(0.0)
      val nevals = state.get[Int]("evalCounter").getOrElse(0)
      config("clr") = -lr / (1 + nevals * lrd)
      state("evalCounter") = nevals + 1
    }
  }

  case class Regime(startEpoch: Int, endEpoch: Int, config: Table)
}
