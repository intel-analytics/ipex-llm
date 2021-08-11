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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{IdentityOutputShape, TensorModule}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, Shape}
import com.intel.analytics.bigdl.utils.RandomGenerator._

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
      if (input.isContiguous()) {
        if (isResampling) {
          val noiseData = noise.storage().array()
          var taskSize = noise.nElement() / Engine.model.getPoolSize
          var extraTask = noise.nElement() % Engine.model.getPoolSize
          var allocated = 0
          val offset = this.output.storageOffset() - 1
          val data = this.output.storage.array()
          var i = 0
          while (allocated < noise.nElement()) {
            val start = allocated
            allocated += taskSize
            if (extraTask > 0) {
              allocated += 1
              extraTask -= 1
            }
            val end = allocated
            results(i) = Engine.model.invoke(() => {
              var k = start
              while (k < end) {
                noiseData(k) = if (RNG.bernoulli(1 - p)) {
                  if (scale) {
                    data(offset + k) = ev.divide(data(offset + k), ev.fromType[Double](1 - p))
                    ev.fromType[Double](1.0 / (1 - p))
                  } else {
                    ev.fromType[Int](1)
                  }
                } else {
                  data(offset + k) = ev.fromType[Int](0)
                  ev.fromType[Int](0)
                }

                k += 1
              }
            })
            i += 1

          }

          Engine.model.sync(results)
        } else {
          this.output.cmul(noise)
        }
        this.output
      } else {
        if (isResampling) {
          noise.bernoulli(1 - p)

          if (scale) {
            noise.div(ev.fromType[Double](1 - p))
          }
        }

        this.output.cmul(noise)
      }
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

      if (gradInput.isContiguous()) {
        val noiseData = noise.storage().array()
        var taskSize = noise.nElement() / Engine.model.getPoolSize
        var extraTask = noise.nElement() % Engine.model.getPoolSize
        val gradInputData = gradInput.storage().array()
        val gradInputOffset = gradInput.storageOffset() - 1
        var allocated = 0
        var i = 0
        while (allocated < noise.nElement()) {
          val start = allocated
          allocated += taskSize
          if (extraTask > 0) {
            allocated += 1
            extraTask -= 1
          }
          val end = allocated
          results(i) = Engine.model.invoke(() => {
            var k = start
            while (k < end) {
              gradInputData(gradInputOffset + k) =
                ev.times(gradInputData(gradInputOffset + k), noiseData(k))
              k += 1
            }
          })
          i += 1
        }

        Engine.model.sync(results)

        this.gradInput
      } else {
        this.gradInput.cmul(noise)
      }
    } else {
      throw new IllegalArgumentException("backprop only defined while training")
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
