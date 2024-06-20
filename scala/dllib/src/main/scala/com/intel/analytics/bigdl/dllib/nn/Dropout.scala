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
package com.intel.analytics.bigdl.dllib.nn

import com.intel.analytics.bigdl.dllib.nn.abstractnn.{IdentityOutputShape, TensorModule}
import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.{Engine, Log4Error, Shape}
import com.intel.analytics.bigdl.dllib.utils.RandomGenerator._

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * Dropout masks(set to zero) parts of input using a bernoulli distribution.
 * Each input element has a probability initP of being dropped. If `scale` is
 * true(true by default), the outputs are scaled by a factor of `1/(1-initP)`
 * during training.
 * During evaluating, output is the same as input.
 *
 * It has been proven an effective approach for regularization and preventing
 * co-adaptation of feature detectors. For more details, plese see
 * [Improving neural networks by preventing co-adaptation of feature detectors]
 * (https://arxiv.org/abs/1207.0580)
 *
 * @param initP the probability p
 * @param inplace whether to make `input` and `output` share the same storage
 * @param scale whether to scale the output by a factor of `1 / (1 - p)`
 */
@SerialVersionUID(- 4636332259181125718L)
class Dropout[T: ClassTag](
  val initP: Double = 0.5,
  val inplace: Boolean = false,
  var scale: Boolean = true)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  private var p = initP
  var noise = Tensor[T]()
  var isResampling = true

  @transient
  protected var results: Array[Future[Unit]] = null

  /**
   * Get current probability to be dropped.
   * @return p
   */
  def getP(): T = {
    return ev.fromType[Double](p)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (inplace) {
      this.output = input
    } else {
      this.output.resizeAs(input).copy(input)
    }

    if (results == null) {
      results = new Array[Future[Unit]](Engine.model.getPoolSize)
    }
    if (train) {
      noise.resizeAs(input)
      if (isResampling) {
        noise.bernoulli(1 - p)

        if (scale) {
          noise.div(ev.fromType[Double](1 - p))
        }
      }

      this.output.cmul(noise)
    } else if (!scale) {
      this.output.mul(ev.fromType[Double](1 - p))
    } else {
      output
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (results == null) {
      results = new Array[Future[Unit]](Engine.model.getPoolSize)
    }
    if (train) {
      if (inplace) {
        this.gradInput = gradOutput
      } else {
        this.gradInput.resizeAs(gradOutput).copy(gradOutput)
      }

      this.gradInput.cmul(noise)
    } else {
      Log4Error.invalidOperationError(false, "Dropout: backprop only defined while training")
    }

    this.gradInput
  }

  override def clearState(): this.type = {
    if (!inplace) {
      super.clearState()
    }
    noise.set()
    this
  }

  /**
   * Set current probability to be dropped.
   * @param p new probability
   * @return
   */
  def setP(p: Double): this.type = {
    this.p = p
    this
  }

  override def toString(): String = {
    s"${getPrintName}($p)"
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    inputShape
  }
}

object Dropout {
  def apply[T: ClassTag](
    initP: Double = 0.5,
    inplace: Boolean = false,
    scale: Boolean = true)(implicit ev: TensorNumeric[T]) : Dropout[T] = {
    new Dropout[T](initP, inplace, scale)
  }
}
