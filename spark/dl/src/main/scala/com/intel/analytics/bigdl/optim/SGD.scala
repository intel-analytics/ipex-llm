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

import com.intel.analytics.bigdl.optim.SGD.{Default, LearningRateSchedule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * A plain implementation of SGD
 * @param learningRate learning rate
 * @param learningRateDecay learning rate decay
 * @param weightDecay weight decay
 * @param momentum momentum
 * @param dampening dampening for momentum
 * @param nesterov enables Nesterov momentum
 * @param learningRates 1D tensor of individual learning rates
 * @param weightDecays 1D tensor of individual weight decays
 * @tparam T
 */
class SGD[@specialized(Float, Double) T: ClassTag](
  var learningRate: Double = 1e-3,
  var learningRateDecay: Double = 0.0,
  var weightDecay: Double = 0.0,
  var momentum: Double = 0.0,
  var dampening: Double = Double.MaxValue,
  var nesterov: Boolean = false,
  var learningRateSchedule: LearningRateSchedule = Default(),
  var learningRates: Tensor[T] = null,
  var weightDecays: Tensor[T] = null)(implicit ev: TensorNumeric[T])
  extends OptimMethod[T] {

  import SGD._

  /**
   *
   * @param feval a function that takes a single input (X), the point of a evaluation,
   * and returns f(X) and df/dX
   * @param x the initial point
   * @return the new x 1D tensor and the function list, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]), x: Tensor[T])
  : (Tensor[T], Array[T]) = {

    this.updateHyperParameter()
    if (this.dampening == Double.MaxValue) this.dampening = this.momentum
    val wd = this.weightDecay
    val mom = this.momentum
    val damp = this.dampening
    val nesterov = this.nesterov
    val lrs = this.learningRates
    val wds = this.weightDecays
    val clr = ev.fromType(this.learningRateSchedule.currentRate)

    require(!nesterov || (mom > 0 && damp == 0),
      "Nesterov momentum requires a momentum and zero dampening")

    var (fx, dfdx) = feval(x)

    if (wd != 0 || wds != null) {
      require(!state.get[Boolean]("isLayerwiseScaled").getOrElse(false),
        "SGD: Can't set layerwise scale and weight decay at the same time")
    }
    if (wd != 0) {
      dfdx.add(ev.fromType[Double](wd), x)
    } else if (wds != null) {
      val decayParameters = state.get[Tensor[T]]("decayParameters").getOrElse({
        val DP = Tensor[T]().resizeAs(dfdx)
        state("decayParameters") = DP
        DP
      })
      decayParameters.copy(wds).cmul(x)
      dfdx.add(decayParameters)
    }

    if (mom != 0) {
      val stateDFDX = state.get[Tensor[T]]("dfdx") match {
        case None =>
          val DFDX = Tensor[T]().resizeAs(dfdx).copy(dfdx)
          state("dfdx") = DFDX
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
    if (lrs != null) {
      val deltaParameters = state.get[Tensor[T]]("deltaParameters").getOrElse({
        val deltaP = Tensor[T]().resizeAs(dfdx)
        state("deltaParameters") = deltaP
        deltaP
      })
      deltaParameters.copy(lrs).cmul(dfdx)
      x.add(clr, deltaParameters)
    } else {
      x.add(clr, dfdx)
    }

    (x, Array(fx))
  }


  override def loadFromTable(config: Table): this.type = {
    this.learningRate = config.get[Double]("learningRate").getOrElse(this.learningRate)
    this.learningRateDecay = config.get[Double]("learningRateDecay")
      .getOrElse(this.learningRateDecay)
    this.weightDecay = config.get[Double]("weightDecay").getOrElse(this.weightDecay)
    this.momentum = config.get[Double]("momentum").getOrElse(this.momentum)
    this.dampening = config.get[Double]("dampening").getOrElse(this.dampening)
    this.nesterov = config.get[Boolean]("nesterov").getOrElse(this.nesterov)
    this.learningRateSchedule = config.get[LearningRateSchedule]("learningRateSchedule")
      .getOrElse(this.learningRateSchedule)
    this.learningRates = config.get[Tensor[T]]("learningRates").getOrElse(this.learningRates)
    this.weightDecays = config.get[Tensor[T]]("weightDecays").getOrElse(this.weightDecays)
    this
  }

  override def clearHistory(): Unit = {
    state.delete("decayParameters")
    state.delete("dfdx")
    state.delete("deltaParameters")
  }

  /**
   * return an string of current hyperParameter.
   */
  override def getHyperParameter(): String = {
    val clr = -this.learningRateSchedule.currentRate
    val wd = this.weightDecay
    val mom = this.momentum
    val damp = this.dampening
    val nesterov = this.nesterov
    val lrs = this.learningRates
    val wds = this.weightDecays
    s"Current learning rate is $clr. " +
      {if (wd != 0) s"Current weight decay is $wd. " else ""} +
      {if (mom != 0) s"Current momentum is $mom. " else ""} +
      {if (damp != 0) s"Current dampening is $damp. " else ""} +
      {if (nesterov) s"Current nesterov is true. " else ""} +
      {if (null != lrs) s"Current learningRates is a Tensor. " else ""} +
      {if (null != wds) s"Current weightDecays is a Tensor. " else ""}
  }

  override def updateHyperParameter(): Unit = {
    this.learningRateSchedule.updateHyperParameter(this)
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

  override def getLearningRate(): Double = this.learningRateSchedule.currentRate
}

object SGD {

  /**
   * Hyper parameter schedule for SGD
   */
  trait LearningRateSchedule {
    /**
     * update learning rate by config table and state table
     * @param optimMethod init optiMethod.
     */
    def updateHyperParameter[T](optimMethod : SGD[T]) : Unit

    @deprecated("Please input SGD instead of Table", "0.2.0")
    def updateHyperParameter(config : Table, state : Table) : Unit = {}

    var currentRate : Double = 0.0
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

    override def updateHyperParameter[T](optimMethod: SGD[T]): Unit = {
      val epoch = optimMethod.state[Int]("epoch")
      for (r <- regimes) {
        if (epoch >= r.startEpoch && epoch <= r.endEpoch) {
          val config = r.config
          val keys = config.keySet.toArray.map(_.toString)
          var i = 0
          while (i < keys.length) {
            keys(i) match {
              case "learningRate" =>
                optimMethod.learningRate = config.get[Double](keys(i)).get
              case "learningRateDecay" =>
                optimMethod.learningRateDecay = config.get[Double](keys(i)).get
              case "weightDecay" =>
                optimMethod.weightDecay = config.get[Double](keys(i)).get
              case "momentum" =>
                optimMethod.momentum = config.get[Double](keys(i)).get
              case "dampening" =>
                optimMethod.dampening = config.get[Double](keys(i)).get
              case "nesterov" =>
                optimMethod.nesterov = config.get[Boolean](keys(i)).get
              case "leaningRateSchedule" =>
                optimMethod.learningRateSchedule = config.get[LearningRateSchedule](keys(i)).get
              case "learningRates" =>
                optimMethod.learningRates = config.get[Tensor[T]](keys(i)).get
              case "weightDecays" =>
                optimMethod.weightDecays = config.get[Tensor[T]](keys(i)).get
              case _ => throw new IllegalArgumentException(
                s"EpochSchedule: ${keys(i)} is not a member of SGD")
            }
            i += 1
          }
        }
      }
      currentRate = -optimMethod.learningRate
    }
  }

  /**
   * A learning rate decay policy, where the effective learning rate
   * follows a polynomial decay, to be zero by the max_iteration.
   * Calculation: base_lr (1 - iter/maxIteration) `^` (power)
   *
   * @param power coeffient of decay, refer to calculation formula
   * @param maxIteration max iteration when lr becomes zero
   */
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

    override def updateHyperParameter[T](optimMethod: SGD[T]): Unit = {
      val lr = optimMethod.learningRate
      val nevals = optimMethod.state.get[Int]("evalCounter").getOrElse(0)
      val clr = if (nevals > maxIteration) {
        0.0
      } else {
        -lr * math.pow(1.0 - nevals.toDouble / maxIteration, power)
      }
      println(s"iteration is : ${nevals}. current learning rate is $clr")
      optimMethod.state("evalCounter") = nevals + 1
      currentRate = clr
    }
  }
  /**
   * A learning rate decay policy, where the effective learning rate
   * is calculated as base_lr * gamma `^` (floor(iter / stepSize))
   *
   * @param stepSize the inteval for lr decay
   * @param gamma coefficient of decay, refer to calculation formula
   */

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

    override def updateHyperParameter[T](optimMethod: SGD[T]): Unit = {
      val lr = optimMethod.learningRate
      var clr = -lr
      val nevals = optimMethod.state.get[Int]("evalCounter").getOrElse(0)
      var i = 0
      while(i < nevals / stepSize) {
        clr *= gamma
        i += 1
      }
      optimMethod.state("evalCounter") = nevals + 1
      currentRate = clr
    }
  }

  /**
   * similar to step but it allows non uniform steps defined by stepSizes
   * @param stepSizes the series of step sizes used for lr decay
   * @param gamma coefficient of decay
   */
  case class MultiStep(stepSizes : Array[Int], gamma : Double) extends LearningRateSchedule {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val lr = config.get[Double]("learningRate").getOrElse(1e-3)
      var clr = -lr
      val nevals = state.get[Int]("evalCounter").getOrElse(0)
      var currentStep = 0
      while (currentStep < stepSizes.length && nevals >= stepSizes(currentStep)) {
        clr *= gamma
        currentStep += 1
      }
      state("evalCounter") = nevals + 1
      config("clr") = clr
    }

    override def updateHyperParameter[T](optimMethod: SGD[T]): Unit = {
      val lr = optimMethod.learningRate
      var clr = -lr
      val nevals = optimMethod.state.get[Int]("evalCounter").getOrElse(0)
      var currentStep = 0
      while (currentStep < stepSizes.length && nevals >= stepSizes(currentStep)) {
        clr *= gamma
        currentStep += 1
      }
      optimMethod.state("evalCounter") = nevals + 1
      currentRate = clr
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

    override def updateHyperParameter[T](optimMethod: SGD[T]): Unit = {
      val lr = optimMethod.learningRate
      var clr = -lr
      val epoch = optimMethod.state[Int]("epoch")
      val decay = decayType(epoch)
      clr = clr * math.pow(0.1, decay)
      currentRate = clr
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

    override def updateHyperParameter[T](optimMethod: SGD[T]): Unit = {
      val lr = optimMethod.learningRate
      var clr = -lr
      val epoch = optimMethod.state[Int]("epoch")
      var i = 0
      while(i < epoch / stepSize) {
        clr *= gamma
        i += 1
      }
      currentRate = clr
    }
  }

  /**
   * [[NaturalExp]] is a learning rate schedule, which rescale the learning rate by
   * exp ( -decay_rate * iter / decay_step )
   * referring to tensorflow's learning rate decay # natural_exp_decay
   *
   * @param decay_step how often to apply decay
   * @param gamma the decay rate. e.g. 0.96
   */
  case class NaturalExp(decay_step : Int, gamma : Double)
    extends LearningRateSchedule {

    override def updateHyperParameter[T](optimMethod: SGD[T]): Unit = {
      val lr = optimMethod.learningRate
      val nevals = optimMethod.state.get[Int]("evalCounter").getOrElse(0)
      val p = nevals / decay_step
      optimMethod.state("evalCounter") = nevals + 1
      val clr = -lr * math.exp(-gamma * p)
      currentRate = clr
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

    override def updateHyperParameter[T](optimMethod: SGD[T]): Unit = {
      val lr = optimMethod.learningRate
      val lrd = optimMethod.learningRateDecay
      val nevals = optimMethod.state.get[Int]("evalCounter").getOrElse(0)
      currentRate = -lr / (1 + nevals * lrd)
      optimMethod.state("evalCounter") = nevals + 1
    }
  }

  /**
   * A structure to specify hyper parameters by start epoch and end epoch.
   * Usually work with [[EpochSchedule]].
   * @param startEpoch start epoch
   * @param endEpoch end epoch
   * @param config config table contains hyper parameters
   */
  case class Regime(startEpoch: Int, endEpoch: Int, config: Table)
}
