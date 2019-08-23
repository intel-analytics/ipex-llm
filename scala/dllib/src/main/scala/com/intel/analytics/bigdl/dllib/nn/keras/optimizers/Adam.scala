/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.pipeline.api.keras.optimizers

import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.optim.SGD.{Default, LearningRateSchedule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.SGDRef

import scala.math._
import scala.reflect.ClassTag


/**
 * An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf with learning rate schedule.
 * @param lr learning rate
 * @param decay learning rate decay
 * @param schedule learning rate schedule
 * @param beta_1 first moment coefficient
 * @param beta_2 second moment coefficient
 * @param epsilon for numerical stability
 */
class Adam[@specialized(Float, Double) T: ClassTag](
    var lr: Double = 1e-3,
    var beta_1: Double = 0.9,
    var beta_2: Double = 0.999,
    var epsilon: Double = 1e-8,
    var decay: Double = 0.0,
    val schedule: LearningRateSchedule = Default()
  )(implicit ev: TensorNumeric[T]) extends SGD[T](learningRate = lr,
    learningRateDecay = decay, learningRateSchedule = schedule) {

  @transient
  private var buffer: Tensor[T] = null

  /**
   * An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   * @return the new x vector and the function list {fx}, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]),
               parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    this.updateHyperParameter()
    if (buffer == null) buffer = Tensor[T]()
    val lr = this.lr
    val lrd = this.decay
    val beta1 = this.beta_1
    val beta2 = this.beta_2
    val eps = this.epsilon

    val (fx, dfdx) = feval(parameter)
    val state = SGDRef.getstate(this)
    var timestep = state.getOrElse[Int]("neval", 0)
    val (_s, _r, _denom) =
      if (state.get[Tensor[T]]("s").isDefined) {
        (state.get[Tensor[T]]("s").get, state.get[Tensor[T]]("r").get,
          state.get[Tensor[T]]("denom").get.resizeAs(dfdx))
      } else {
        (Tensor[T]().resizeAs(dfdx).zero(), Tensor[T]().resizeAs(dfdx).zero(),
          Tensor[T]().resizeAs(dfdx).zero())
      }

    val clr = - this.schedule.currentRate

    /**
     * m_t = beta_1 * m_t-1 + (1 - beta_1) * g_t
     * v_t = beta_2 * v_t-1 + (1 - beta_2) * g_t * g_t
     */
    _s.mul(ev.fromType[Double](beta1)).add(ev.fromType[Double](1-beta1), dfdx)
    // buffer = dfdx * dfdx
    buffer.resizeAs(dfdx).cmul(dfdx, dfdx)
    _r.mul(ev.fromType[Double](beta2)).add(ev.fromType[Double](1-beta2), buffer)
    _denom.sqrt(_r)

    // used as MKL.axpy: 1 * a + y = y, and fill buffer with one
    buffer.fill(ev.one)
    _denom.add(ev.fromType(eps), buffer)

    // efficiency improved upon by changing the order of computation, at expense of clarity
    val biasCorrection1 = 1 - pow(beta1, timestep)
    val biasCorrection2 = 1 - pow(beta2, timestep)
    val stepSize = clr * sqrt(biasCorrection2) / biasCorrection1
    parameter.addcdiv(ev.fromType[Double](-stepSize), _s, _denom)

    state("s") = _s // 1st moment variables
    state("r") = _r // 2nd moment variables
    state("denom") = _denom // 3nd moment variables

    (parameter, Array(fx))
  }

  override def loadFromTable(config: Table): this.type = {
    super.loadFromTable(config)
    this.beta_1 = config.get[Double]("beta1").getOrElse(this.beta_1)
    this.beta_2 = config.get[Double]("beta2").getOrElse(this.beta_2)
    this.epsilon = config.get[Double]("Epsilon").getOrElse(this.epsilon)
    this
  }

  override def clearHistory(): Unit = {
    super.clearHistory()
    val state = SGDRef.getstate(this)
    state.delete("s")
    state.delete("r")
  }
}


/**
 * A learning rate decay policy, where the effective learning rate
 * follows a polynomial decay, to be zero by the max_epochs.
 * Calculation: init_lr * (1 - epoch/max_iteration) ^ (power)
 *
 * @param power The coefficient of decay.
 * @param maxEpochs The maximum number of epochs when lr becomes zero.
 */
case class PolyEpochDecay(power: Double, maxEpochs: Int) extends LearningRateSchedule {
  override def updateHyperParameter[T](optimMethod: SGD[T]): Unit = {
    val state = SGDRef.getstate(optimMethod)
    val epoch = state[Int]("epoch")
    val lr = optimMethod.learningRate
    val clr = if (epoch >= maxEpochs) {
      0.0
    } else {
      -lr * math.pow(1.0 - epoch.toDouble / maxEpochs, power)
    }
    currentRate = clr
  }
}
