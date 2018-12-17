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

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * This class is intended to support inputs with 3 or more dimensions.
 * Apply Any Provided Criterion to every temporal slice of an input.
 *
 * @param critrn embedded criterion
 * @param sizeAverage whether to divide the length of input along dimension.
 * @param dimension to compute criterion loss along the input and target dimension
 *                  if sizeAverage=true, and dimension=1, it means to divide the batch size;
 *                  if sizeAverage=true, and dimension=2, it means to divide the sequence length
 */

class TimeDistributedCriterion[T : ClassTag](
  val critrn : TensorCriterion[T],
  val sizeAverage: Boolean = false,
  val dimension: Int = 2)
  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  private var fInput: Tensor[T] = Tensor[T]()
  private var fTarget: Tensor[T] = Tensor[T]()
  private var _gradInput = Tensor[T]()  // list of cell criterions cloned from added criterion
  private val cells: ArrayBuffer[TensorCriterion[T]]
  = ArrayBuffer[TensorCriterion[T]]()

  @transient
  protected var results: Array[Future[Unit]] = _

  /**
   * Clone N criterions; N depends on the time dimension of the input
   * @param times
   */
  private def extend(times: Int): Unit = {
    var t = cells.length
    while (t < times) {
      cells += critrn.cloneCriterion()
        .asInstanceOf[TensorCriterion[T]]
      t += 1
    }
  }


  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    /**
     * Take each time slice of input and target, and add up all outputs of slices
     * Example with dimension=2:
     * input.size = [B, T, D] => fInput.size = [B, D]
     * target.size = [B, T] => fTarget.size = [B]
     * If sizeAverage is true, the output is averaged through time dim
     */
    require(input.size(dimension) == target.size(dimension),
      "target should have as many elements as input, " +
        s"input ${input.size(dimension)}, target ${target.size(dimension)}")

    output = ev.fromType[Int](0)
    val nstep = input.size(dimension)
    extend(nstep)

    if (results == null || results.length != nstep) {
      results = new Array[Future[Unit]](nstep)
    }

    var i = 0
    while (i < nstep) {
      val _i = i + 1
      results(i) = Engine.model.invoke(() => {
        fInput = input.select(dimension, _i)
        fTarget = target.select(dimension, _i)
        cells(_i - 1).updateOutput(fInput, fTarget)
      })
      i += 1
    }
    Engine.model.sync(results)

    (0 until nstep).foreach(b => {
      output = ev.plus(output, cells(b).output)
    })

    if (sizeAverage) {
      output = ev.divide(output, ev.fromType[Int](nstep))
    }
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    /**
     * Take each time slice of input and target, and calculate gradInput of each slice
     * If sizeAverage is true, the gradInput is also averaged through dimension
     */
    require(input.size(dimension) == target.size(dimension),
      s"target should have as many elements as input, " +
        s"input ${input.size(dimension)}, target ${target.size(dimension)}")
    gradInput.resizeAs(input).zero()

    val nstep = input.size(dimension)

    var i = 0
    while (i < nstep) {
      val _i = i + 1
      results(i) = Engine.model.invoke(() => {
        fInput = input.select(dimension, _i)
        fTarget = target.select(dimension, _i)
        _gradInput = gradInput.select(dimension, _i)
        _gradInput.copy(cells(_i - 1).updateGradInput(fInput, fTarget).toTensor[T])
        if (sizeAverage) {
          _gradInput = _gradInput.div(ev.fromType[Int](nstep))
        }
      })
      i += 1
    }
    Engine.model.sync(results)
    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[TimeDistributedCriterion[T]]
}

object TimeDistributedCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
    critrn: TensorCriterion[T] = null, sizeAverage: Boolean = false, dimension: Int = 2)
    (implicit ev: TensorNumeric[T]) : TimeDistributedCriterion[T] = {
    new TimeDistributedCriterion[T](critrn, sizeAverage, dimension)
  }
}
