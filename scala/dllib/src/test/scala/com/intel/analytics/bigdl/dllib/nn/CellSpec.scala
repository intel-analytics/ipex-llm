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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.ClassTag

private[bigdl] class CellUnit[T : ClassTag] (hidSize: Int)
  (implicit ev: TensorNumeric[T])
  extends Cell[T](hiddensShape = Array(hidSize, hidSize, hidSize)) {

  override def updateOutput(input: Table): Table = {
    T()
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    T()
  }

  override def accGradParameters(input: Table, gradOutput: Table): Unit = {}

  override var cell: AbstractModule[Activity, Activity, T] = _
  override var preTopology: TensorModule[T] = null
}

@com.intel.analytics.bigdl.tags.Parallel
class CellSpec extends FlatSpec with Matchers {

  "A Cell" should "hidResize correctly" in {
    val cell = new CellUnit[Double](4)
    val stepShape = Array(1)
    val hidden = cell.hidResize(hidden = null, batchSize = 5, stepShape)

    hidden.isInstanceOf[Table] should be (true)
    var i = 1
    while (i < hidden.toTable.length) {
      hidden.toTable(i).asInstanceOf[Tensor[Double]].size should be (Array(5, 4))
      i += 1
    }

    val hidden2 = T(Tensor[Double](3, 4), Tensor[Double](4, 5), Tensor[Double](5, 6))
    cell.hidResize(hidden2, 5, stepShape)
    hidden2(1).asInstanceOf[Tensor[Double]].size should be (Array(5, 4))
    hidden2(2).asInstanceOf[Tensor[Double]].size should be (Array(5, 4))
    hidden2(3).asInstanceOf[Tensor[Double]].size should be (Array(5, 4))

  }
}
