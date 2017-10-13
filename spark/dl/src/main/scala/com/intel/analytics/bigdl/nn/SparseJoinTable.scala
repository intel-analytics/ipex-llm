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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.{DenseTensor, SparseTensor, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, Table}

import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * :: Experimental ::
 *
 * Sparse version of JoinTable. Backward just pass the origin gradOutput back to
 * the next layers without split. So this layer may just works in Wide&Deep like models.
 *
 * @param dimension the dimension to join.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
class SparseJoinTable[T: ClassTag] (
    val dimension: Int)(implicit ev: TensorNumeric[T])
    extends AbstractModule[Table, Tensor[T], T] {

  private var results: Array[Future[Unit]] = null
  output = Tensor.sparse(Array(1, 1), 1)

  var size: Array[Int] = null

  override def updateOutput(input: Table): Tensor[T] = {
    var nElements = 0

    var i = 1
    while (i <= input.length()) {
      val currentOutput: Tensor[T] = input(i)
      if (i == 1) {
        size = currentOutput.size()
      } else {
        size(dimension - 1) += currentOutput.size(dimension)
      }
      nElements += currentOutput.nElement()
      i += 1
    }
    output.resize(size, nElements)

    Tensor.sparseConcat(2, input, output)

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    var i = 1
    while (i <= input.length()) {
      gradInput(i) = gradOutput
      i += 1
    }
    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    size = null
    results = null
    this
  }

  override def toString: String = s"nn.SparseJoinTable($dimension)"


  override def canEqual(other: Any): Boolean = other.isInstanceOf[SparseJoinTable[T]]

  override def equals(other: Any): Boolean = other match {
    case that: SparseJoinTable[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        dimension == that.dimension
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), dimension)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object SparseJoinTable {
  def apply[@specialized(Float, Double) T: ClassTag](
        dimension: Int)(implicit ev: TensorNumeric[T]) : SparseJoinTable[T] = {
    new SparseJoinTable[T](dimension)
  }
}
