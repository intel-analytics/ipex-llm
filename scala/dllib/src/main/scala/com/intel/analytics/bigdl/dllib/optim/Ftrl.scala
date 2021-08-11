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

import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc6}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * An implementation of Ftrl https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf.
 * Support L1 penalty, L2 penalty and shrinkage-type L2 penalty.
 *
 * @param learningRate learning rate
 * @param learningRatePower double, must be less or equal to zero. Default is -0.5.
 * @param initialAccumulatorValue double, the starting value for accumulators,
 *     require zero or positive values. Default is 0.1.
 * @param l1RegularizationStrength double, must be greater or equal to zero. Default is zero.
 * @param l2RegularizationStrength double, must be greater or equal to zero. Default is zero.
 * @param l2ShrinkageRegularizationStrength double, must be greater or equal to zero.
 *     Default is zero. This differs from l2RegularizationStrength above. L2 above is a
 *     stabilization penalty, whereas this one is a magnitude penalty.
 */
class Ftrl[@specialized(Float, Double) T: ClassTag](
  var learningRate: Double = 1e-3,
  var learningRatePower: Double = -0.5,
  var initialAccumulatorValue: Double = 0.1,
  var l1RegularizationStrength: Double = 0.0,
  var l2RegularizationStrength: Double = 0.0,
  var l2ShrinkageRegularizationStrength: Double = 0.0
  )(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {

  @transient var accumNew: Tensor[T] = _
  @transient var buffer: Tensor[T] = _
  @transient var quadratic: Tensor[T] = _
  @transient var gradWithStrinkage: Tensor[T] = _

  protected def checkParam(learningRate: Double,
         learningRatePower: Double,
         initialAccumulatorValue: Double,
         l1RegularizationStrength: Double,
         l2RegularizationStrength: Double,
         l2ShrinkageRegularizationStrength: Double): Unit = {
    require(learningRate >= 0.0, s"Ftrl: learning rate should be greater or equal to zero." +
      s" but got $learningRate")
    require(learningRatePower <= 0.0,
      s"Ftrl: learning rate power should be smaller or equal to zero." +
        s" but got $learningRatePower")
    require(initialAccumulatorValue >= 0.0,
      s"Ftrl: initial value of accumulator should be greater or equal to zero." +
        s" but got $initialAccumulatorValue")
    require(l1RegularizationStrength >= 0.0,
      s"Ftrl: L1 regularization strength should be greater or equal to zero." +
        s" but got $l1RegularizationStrength")
    require(l2RegularizationStrength >= 0.0,
      s"Ftrl: L2 regularization strength should be greater or equal to zero." +
        s" but got $l2RegularizationStrength")
    require(l2ShrinkageRegularizationStrength >= 0.0,
      s"Ftrl: L2 shrinkage regularization strength should be greater or equal to zero." +
        s" but got $l2ShrinkageRegularizationStrength")
  }

  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]),
               parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    checkParam(learningRate, learningRatePower, initialAccumulatorValue, l1RegularizationStrength,
      l2RegularizationStrength, l2ShrinkageRegularizationStrength)
    val lr = this.learningRate
    val lrp = this.learningRatePower
    val iav = ev.fromType(this.initialAccumulatorValue)
    val l1rs = ev.fromType(this.l1RegularizationStrength)
    val l2rs = ev.fromType(this.l2RegularizationStrength)
    val l2srs = ev.fromType(this.l2ShrinkageRegularizationStrength)

    val (fx, dfdx) = feval(parameter)

    val (accum, linear) = if (state.get[Tensor[T]]("accum").isDefined) {
      (state.get[Tensor[T]]("accum").get, state.get[Tensor[T]]("linear").get)
    } else {
      // fill accum with initialAccumulatorValue
      (Tensor[T]().resizeAs(dfdx).fill(iav), Tensor[T]().resizeAs(dfdx))
    }

    if (accumNew == null || !accumNew.isSameSizeAs(dfdx)) {
      accumNew = Tensor[T]().resizeAs(dfdx).copy(accum)
    }

    if (buffer == null || !buffer.isSameSizeAs(dfdx)) buffer = Tensor[T]().resizeAs(dfdx)

    if (quadratic == null || !quadratic.isSameSizeAs(dfdx)) quadratic = Tensor[T]().resizeAs(dfdx)

    if (gradWithStrinkage == null || !gradWithStrinkage.isSameSizeAs(dfdx)) {
      gradWithStrinkage = Tensor[T]().resizeAs(dfdx)
    }

    val computeParameter = new TensorFunc6[T]() {
      // parameter = (sign(linear) * l1rs - linear) / quadratic if |linear| > l1rs else 0.0
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
                         data3: Array[T], offset3: Int): Unit = {
        data1(offset1) = if (ev.isGreater(ev.abs(data2(offset2)), l1rs)) {
          val l1 = if (ev.isGreater(data2(offset2), ev.zero)) {
            l1rs
          } else if (ev.isGreater(ev.zero, data2(offset2))) {
            ev.negative(l1rs)
          } else {
            ev.zero
          }
          ev.divide(ev.minus(l1, data2(offset2)), data3(offset3))
        } else {
          ev.zero
        }
      }
    }

    if (ev.isGreaterEq(ev.zero, l2srs)) {
      // accumNew = accum + dfdx * dfdx
      accumNew.addcmul(dfdx, dfdx)
      // linear += dfdx + accum^(-lrp) / lr * parameter - accumNew^(-lrp) / lr * parameter
      linear.add(dfdx)
      buffer.pow(accum, ev.fromType(-lrp))
      linear.addcmul(ev.fromType(1 / lr), buffer, parameter)
      buffer.pow(accumNew, ev.fromType(-lrp))
      linear.addcmul(ev.fromType(-1 / lr), buffer, parameter)
      // quadratic = 1.0 / lr * accumNew^(- lrp) + 2 * l2
      quadratic.fill(ev.times(ev.fromType(2), l2rs))
      quadratic.add(ev.fromType(1 / lr), buffer)
      // parameter = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
      DenseTensorApply.apply3(parameter, linear, quadratic, computeParameter)
    } else {
      // gradWithShrinkage = dfdx + 2 * l2srs * parameter
      gradWithStrinkage.copy(dfdx)
      gradWithStrinkage.add(ev.times(ev.fromType(2), l2srs), parameter)
      // accumNew = accum + gradWithShrinkage * gradWithShrinkage
      accumNew.addcmul(gradWithStrinkage, gradWithStrinkage)
      // linear += gradWithStrinkage + accum^(-lrp) / lr * parameter
      // - accumNew^(-lrp) / lr * parameter
      linear.add(gradWithStrinkage)
      buffer.pow(accum, ev.fromType(-lrp))
      linear.addcmul(ev.fromType(1.0 / lr), buffer, parameter)
      buffer.pow(accumNew, ev.fromType(-lrp))
      linear.addcmul(ev.fromType(-1.0 / lr), buffer, parameter)
      // quadratic = 1.0 / lr * accumNew^(- lrp) + 2 * l2
      quadratic.fill(ev.times(ev.fromType(2), l2rs))
      quadratic.add(ev.fromType(1 / lr), buffer)
      // parameter = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
      DenseTensorApply.apply3(parameter, linear, quadratic, computeParameter)
    }
    // accum = accum_new
    accum.copy(accumNew)

    state("accum") = accum
    state("linear") = linear

    (parameter, Array(fx))
  }

  override def loadFromTable(config: Table): this.type = {
    this.learningRate = config.get[Double]("learningRate").getOrElse(this.learningRate)
    this.learningRatePower = config.get[Double]("learningRatePower")
      .getOrElse(this.learningRatePower)
    this.initialAccumulatorValue = config.get[Double]("initialAccumulatorValue")
      .getOrElse(this.initialAccumulatorValue)
    this.l1RegularizationStrength = config.get[Double]("l1RegularizationStrength")
      .getOrElse(this.l1RegularizationStrength)
    this.l2RegularizationStrength = config.get[Double]("l2RegularizationStrength")
      .getOrElse(this.l2RegularizationStrength)
    this.l2ShrinkageRegularizationStrength = config.get[Double]("l2ShrinkageRegularizationStrength")
      .getOrElse(this.l2ShrinkageRegularizationStrength)
    this
  }

  override def clearHistory(): Unit = {
    state.delete("accum")
    state.delete("linear")
    accumNew = null
    buffer = null
    quadratic = null
  }

  override def getLearningRate(): Double = this.learningRate
}
