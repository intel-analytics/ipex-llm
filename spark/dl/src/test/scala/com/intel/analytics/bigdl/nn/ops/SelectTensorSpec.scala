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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}


class SelectTensorSpec extends FlatSpec with Matchers  {

  private val t1 = Tensor[Double](2, 3).randn()
  private val t2 = Tensor[Double](2, 3).randn()
  private val t3 = Tensor[Double](2, 3).randn()
  private val t4 = Tensor[Double](2, 3).randn()
  private val t5 = Tensor[Double](2, 3).randn()
  private val table = T()
    .update(Tensor.scalar(1), t1)
    .update(Tensor.scalar("2"), t2)
    .update(3, t3)
    .update(4.0f, t4)
    .update("5", t5)

  "SelectedTensor with TensorKey" should "work correctly" in {
    val t1Copy = SelectTensor[Double](Tensor.scalar(1)).forward(table)
    t1Copy shouldEqual t1
    val t1Values = t1.storage().array().clone()
    t1Copy.square()
    t1.storage().array() shouldEqual t1Values
    t1Copy.storage().array() shouldEqual t1Values.map(e => e * e)

    val t2Copy = SelectTensor[Double](Tensor.scalar("2"), true).forward(table)
    t2Copy shouldEqual t2
  }

  "SelectedTensor with primitive Key" should "work correctly" in {
    val t3Copy = SelectTensor[Double, Int](3).forward(table)
    t3Copy shouldEqual t3

    val t4Copy = SelectTensor[Double](Tensor.scalar(4.0f), false).forward(table)
    t4Copy shouldEqual t4

    val t5Copy = SelectTensor[Double, String]("5").forward(table)
    t5Copy shouldEqual t5
  }

  "SelectedTensor with transformer" should "work correctly" in {
    val transformer = (TensorOp[Double]() ** 3 * 4.5).ceil
    var select = SelectTensor(Tensor.scalar("2"), transformer = transformer)
    val t2Values = t2.storage().array().clone()
    val t2Convert = select.forward(table)
    t2Convert.storage().array() shouldEqual
      t2Values.map(e => math.ceil(math.pow(e, 3) * 4.5))

    val transformer2 = TensorOp[Double]().abs.ceil.inv * 3.0
    select = SelectTensor(Tensor.scalar("5"), false, transformer = transformer2)
    val t5Values = t5.storage().array().clone()
    val t5Convert = select.forward(table)
    t5Convert.storage().array() shouldEqual
      t5Values.map(e => 3.0 / math.ceil(math.abs(e)))
  }

}

class SelectTensorSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val transformer = (TensorOp[Float]() ** 3 * 4.5f).ceil
    val select = SelectTensor(Tensor.scalar("2"), transformer = transformer)
    val t1 = Tensor[Float](3, 4).randn()
    val t2 = Tensor[Float](2, 3).randn()
    val input = T().update(Tensor.scalar(1), t1).update(Tensor.scalar("2"), t2)
    runSerializationTest(select, input)
  }
}
