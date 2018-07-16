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
import com.intel.analytics.bigdl.utils.{Engine, T, Table}

import scala.math._
import scala.reflect.ClassTag

/**
 * An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf
 * @param learningRate learning rate
 * @param learningRateDecay learning rate decay
 * @param beta1 first moment coefficient
 * @param beta2 second moment coefficient
 * @param Epsilon for numerical stability
 * @tparam T
 */
class Adam[@specialized(Float, Double) T: ClassTag](
    var learningRate: Double = 1e-3,
    var learningRateDecay: Double = 0.0,
    var beta1: Double = 0.9,
    var beta2: Double = 0.999,
    var Epsilon: Double = 1e-8)(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {

  @transient
  private var ones: Tensor[T] = null

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
    val lr = this.learningRate
    val lrd = this.learningRateDecay
    val beta1 = this.beta1
    val beta2 = this.beta2
    val eps = this.Epsilon

    val (fx, dfdx) = feval(parameter)

    var timestep = state.getOrElse[Int]("evalCounter", 0)

    val clr = lr / (1 + timestep*lrd)

    timestep = timestep + 1

    val parallelNum = Engine.coreNumber()
    val gradLength = parameter.nElement()
    val taskSize = gradLength / parallelNum
    val extraTask = gradLength % parallelNum
    if (ones == null || ones.nElement() < taskSize + 1) {
      ones = Tensor[T]().resize(taskSize + 1).fill(ev.one)
    }

    Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
      val offset = tid * taskSize + math.min(tid, extraTask)
      val length = taskSize + (if (tid < extraTask) 1 else 0)
      val currentDfdx = dfdx.narrow(1, offset + 1, length)
      val currentParameter = parameter.narrow(1, offset + 1, length)
      val currentOnes = ones.narrow(1, 1, length)
      val (_s, _r, _denom) =
        if (state.get[Tensor[T]](s"s$tid").isDefined && state.get[Tensor[T]](s"r$tid").isDefined
          && state.get[Tensor[T]](s"denom$tid").isDefined) {
          (state.get[Tensor[T]](s"s$tid").get, state.get[Tensor[T]](s"r$tid").get,
            state.get[Tensor[T]](s"denom$tid").get)
        } else {
          (Tensor[T]().resizeAs(currentParameter).zero(),
            Tensor[T]().resizeAs(currentParameter).zero(),
            Tensor[T]().resizeAs(currentParameter).zero())
        }
      Adam.updateFrame(_s, _r, _denom, clr, currentDfdx, currentParameter,
        beta1, beta2, timestep, currentOnes, eps)

      state(s"s$tid") = _s // 1st moment variables
      state(s"r$tid") = _r // 2nd moment variables
      state(s"denom$tid") = _denom // 3nd moment variables
    }))


    state("evalCounter") = timestep // A tmp tensor to hold the sqrt(v) + epsilon

    (parameter, Array(fx))
  }

  override def loadFromTable(config: Table): this.type = {
    this.learningRate = config.get[Double]("learningRate").getOrElse(this.learningRate)
    this.learningRateDecay = config.get[Double]("learningRateDecay")
      .getOrElse(this.learningRateDecay)
    this.beta1 = config.get[Double]("beta1").getOrElse(this.beta1)
    this.beta2 = config.get[Double]("beta2").getOrElse(this.beta2)
    this.Epsilon = config.get[Double]("Epsilon").getOrElse(this.Epsilon)
    this
  }

  override def clearHistory(): Unit = {
    state.delete("s")
    state.delete("r")
  }

  override def getLearningRate(): Double = this.learningRate
}

object Adam {
  private[optim] def updateFrame[T: ClassTag](_s: Tensor[T], _r: Tensor[T], _denom: Tensor[T],
                                              clr: Double, dfdx: Tensor[T], parameter: Tensor[T],
                                              beta1: Double, beta2: Double, timestep: Int,
                                              ones: Tensor[T], eps: Double)(
                                                 implicit ev: TensorNumeric[T]): Unit = {
    /**
     * m_t = beta_1 * m_t-1 + (1 - beta_1) * g_t
     * v_t = beta_2 * v_t-1 + (1 - beta_2) * g_t * g_t
     */
    _s.mul(ev.fromType[Double](beta1)).add(ev.fromType[Double](1-beta1), dfdx)
    _r.mul(ev.fromType[Double](beta2)).addcmul(ev.fromType[Double](1-beta2), dfdx, dfdx)
    _denom.sqrt(_r)

    // used as MKL.axpy: 1 * a + y = y, and fill buffer with one
    _denom.add(ev.fromType(eps), ones)

    // efficiency improved upon by changing the order of computation, at expense of clarity
    val biasCorrection1 = 1 - pow(beta1, timestep)
    val biasCorrection2 = 1 - pow(beta2, timestep)
    val stepSize = clr * sqrt(biasCorrection2) / biasCorrection1
    parameter.addcdiv(ev.fromType[Double](-stepSize), _s, _denom)
  }
}
