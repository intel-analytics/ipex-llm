/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator}
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag

class Dropout[@specialized(Float, Double) T: ClassTag](
  val initP: Double = 0.5, val inplace: Boolean = false, var scale: Boolean = true)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  private var p = initP
  val noise = Tensor[T]()

  @transient
  protected var results: Array[Future[Unit]] = null

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
      results = new Array[Future[Unit]](Engine.coresNum)
    }
    if (train) {
      noise.resizeAs(input)
      if (input.isContiguous()) {
        val noiseData = noise.storage().array()
        var taskSize = noise.nElement() / Engine.coresNum
        var extraTask = noise.nElement() % Engine.coresNum
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
          results(i) = Future {
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
          }(Engine.getInstance())
          i += 1
        }

        val allocatedTask = i
        i = 0
        while (i < allocatedTask) {
          Await.result(results(i), Duration.Inf)
          i += 1
        }
        this.output
      } else {
        noise.bernoulli(1 - p)

        if (scale) {
          noise.div(ev.fromType[Double](1 - p))
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
      results = new Array[Future[Unit]](Engine.coresNum)
    }
    if (train) {
      if (inplace) {
        this.gradInput = gradOutput
      } else {
        this.gradInput.resizeAs(gradOutput).copy(gradOutput)
      }

      if (gradInput.isContiguous()) {
        val noiseData = noise.storage().array()
        var taskSize = noise.nElement() / Engine.coresNum
        var extraTask = noise.nElement() % Engine.coresNum
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
          results(i) = Future {
            var k = start
            while (k < end) {
              gradInputData(gradInputOffset + k) =
                ev.times(gradInputData(gradInputOffset + k), noiseData(k))
              k += 1
            }
          }(Engine.getInstance())
          i += 1
        }

        val allocatedTask = i
        i = 0
        while (i < allocatedTask) {
          Await.result(results(i), Duration.Inf)
          i += 1
        }

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
    super.clearState()
    noise.set()
    this
  }

  def setP(p: Double): this.type = {
    this.p = p
    this
  }

  override def toString(): String = {
    s"nn.Dropout($p)"
  }
}

object Dropout {
  def apply[@specialized(Float, Double) T: ClassTag](
      initP: Double = 0.5,
      inplace: Boolean = false,
      scale: Boolean = true)(implicit ev: TensorNumeric[T]) : Dropout[T] = {
    new Dropout[T](initP, inplace, scale)
  }
}
