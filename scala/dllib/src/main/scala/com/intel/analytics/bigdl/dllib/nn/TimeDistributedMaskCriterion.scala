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

import com.intel.analytics.bigdl.nn.abstractnn.{SizeAverageStatus, TensorCriterion}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * This class is intended to support inputs with 3 or more dimensions.
 * Apply Any Provided Criterion to every temporal slice of an input.
 * In addition, it supports padding mask.
 *
 * eg. if the target is [ [-1, 1, 2, 3, -1], [5, 4, 3, -1, -1] ],
 *     and set the paddingValue property to -1, then the loss of -1 would not
 *     be accumulated and the loss is only divided by 6 (ont including the amount of
 *     -1, in this case, we are only interested in 1, 2, 3, 5, 4, 3)
 *
 * @param critrn embedded criterion
 * @param paddingValue padding value
 */

class TimeDistributedMaskCriterion[T : ClassTag](
  val critrn : TensorCriterion[T],
  val paddingValue: Int = 0
)
  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  val dimension: Int = 2
  private var fInput: Tensor[T] = Tensor[T]()
  private var fTarget: Tensor[T] = Tensor[T]()
  private var _gradInput = Tensor[T]()  // list of cell criterions cloned from added criterion
  private val cells: ArrayBuffer[TensorCriterion[T]]
  = ArrayBuffer[TensorCriterion[T]]()

  @transient
  protected var results: Array[Future[Unit]] = _
  private val mask = Tensor[T]()
  private val sumBuffer = Tensor[T]()
  private val gradInputBuffer = Tensor[T]()

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

    mask.resizeAs(target)
    mask.applyFun[T](target, x =>
      if (x != ev.fromType[Int](paddingValue)) ev.one else ev.zero)

    sumBuffer.sum(mask, dimension % 2 + 1)
    var sum = ev.zero
    (0 until nstep).foreach(t => {
      val loss = critrn.sizeAverageStatus match {
        case SizeAverageStatus.True =>
          ev.times(cells(t).output, sumBuffer(Array(1, t + 1)))
        case SizeAverageStatus.False => cells(t).output
        case SizeAverageStatus.None =>
          throw new RuntimeException("Using TimeDistributedMaskCriterion," +
            " the embedded criterion should be set to True or False")
      }
      sum = ev.plus(sum, loss)
    })

    output = ev.divide(sum, mask.sum())

    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    /**
     * Take each time slice of input and target, and calculate gradInput of each slice
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
        val _iGradInput = cells(_i - 1).updateGradInput(fInput, fTarget).toTensor[T]
        _gradInput.copy(
          critrn.sizeAverageStatus match {
            case SizeAverageStatus.True =>
              gradInputBuffer.resizeAs(_iGradInput).mul(
                _iGradInput,
                sumBuffer(Array(1, _i)))
            case SizeAverageStatus.False => _iGradInput
            case SizeAverageStatus.None =>
              throw new RuntimeException("Using TimeDistributedMaskCriterion," +
                " the embedded criterion should be set to True or False")
          })
      })
      i += 1
    }
    Engine.model.sync(results)
    gradInput.div(mask.sum())
    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[TimeDistributedCriterion[T]]
}

object TimeDistributedMaskCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
    critrn: TensorCriterion[T] = null,
    paddingValue: Int = 0
  )
    (implicit ev: TensorNumeric[T]) : TimeDistributedMaskCriterion[T] = {
    new TimeDistributedMaskCriterion[T](critrn, paddingValue)
  }
}
