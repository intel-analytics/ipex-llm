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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{RandomGenerator, T}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class SparseJoinTableSpec  extends FlatSpec with Matchers {

  "Sparse JoinTable" should "return the same result" in {
    Random.setSeed(2)
    RandomGenerator.RNG.setSeed(1)
    val input = Tensor(4, 3).apply1(_ => Random.nextInt(2) * Random.nextFloat())
    println(input)
    val input2 = Tensor(4, 2).apply1(_ => Random.nextInt(2) * Random.nextFloat())
    println(input2)
    val sparseModel = Sequential().add(ParallelTable().add(Identity()).add(Identity()))
      .add(SparseJoinTable(2))
    val denseInput = Tensor(4, 5)
    denseInput.narrow(2, 1, 3).copy(input)
    denseInput.narrow(2, 4, 2).copy(input2)

    val sparseInput = T(Tensor.sparse(input), Tensor.sparse(input2))
    val out1 = sparseModel.forward(sparseInput).toTensor[Float]
    val exceptOut = Tensor.sparse(denseInput)
    out1 shouldEqual exceptOut
    Tensor.dense(out1) shouldEqual denseInput

  }

  "Sparse JoinTable" should "return the same result 2" in {
    Random.setSeed(2)
    RandomGenerator.RNG.setSeed(1)
    val input = Tensor(4, 10).apply1(_ => Random.nextInt(10) / 9 * Random.nextFloat())
    println(input)
    val input2 = Tensor(4, 10).apply1(_ => Random.nextInt(10) / 9 * Random.nextFloat())
    println(input2)
    val sparseModel = Sequential().add(ParallelTable().add(Identity()).add(Identity()))
      .add(SparseJoinTable(2))
    val denseInput = Tensor(4, 20)
    denseInput.narrow(2, 1, 10).copy(input)
    denseInput.narrow(2, 11, 10).copy(input2)

    val sparseInput = T(Tensor.sparse(input), Tensor.sparse(input2))
    val out1 = sparseModel.forward(sparseInput).toTensor[Float]
    val exceptOut = Tensor.sparse(denseInput)
    out1 shouldEqual exceptOut
    Tensor.dense(out1) shouldEqual denseInput

  }

  "Sparse JoinTable on narrowed table" should "return the same result" in {
    Random.setSeed(2)
    RandomGenerator.RNG.setSeed(1)
    val input = Tensor(8, 10).apply1(_ => Random.nextInt(10) / 5 * Random.nextFloat())
    println(input)
    val input2 = Tensor(4, 10).apply1(_ => Random.nextInt(10) / 5 * Random.nextFloat())
    println(input2)
    val sparseModel = Sequential().add(ParallelTable().add(Identity()).add(Identity()))
      .add(SparseJoinTable(2))
    val denseInput = Tensor(4, 20)
    denseInput.narrow(2, 1, 10).copy(input.narrow(1, 4, 4))
    denseInput.narrow(2, 11, 10).copy(input2)

    val sparseInput = T(Tensor.sparse(input).narrow(1, 4, 4), Tensor.sparse(input2))
    val out1 = sparseModel.forward(sparseInput).toTensor[Float]
    val exceptOut = Tensor.sparse(denseInput)
    out1 shouldEqual exceptOut
    Tensor.dense(out1) shouldEqual denseInput
  }

  "Sparse JoinTable on narrowed table" should "return the same result 2" in {
    Random.setSeed(2)
    RandomGenerator.RNG.setSeed(1)
    val input = Tensor(4, 10).apply1(_ => Random.nextInt(10) / 5 * Random.nextFloat())
    val input2 = Tensor(8, 10).apply1(_ => Random.nextInt(10) / 5 * Random.nextFloat())
    val sparseModel = Sequential().add(ParallelTable().add(Identity()).add(Identity()))
      .add(SparseJoinTable(2))
    val denseInput = Tensor(4, 20)
    denseInput.narrow(2, 1, 10).copy(input)
    denseInput.narrow(2, 11, 10).copy(input2.narrow(1, 4, 4))

    val sparseInput = T(Tensor.sparse(input), Tensor.sparse(input2).narrow(1, 4, 4))
    val out1 = sparseModel.forward(sparseInput).toTensor[Float]
    val exceptOut = Tensor.sparse(denseInput)
    out1 shouldEqual exceptOut
    Tensor.dense(out1) shouldEqual denseInput
  }

  "Sparse JoinTable on narrowed table" should "return the same result 3" in {
    val indices1 = Array(0, 1, 2, 3, 3)
    val indices2 = Array(0, 1, 2, 3, 4)
    val values1 = Array(1f, 2f, 3f, 4f, 5f)
    val input1 = Tensor.sparse(Array(indices1, indices2), values1, Array(4, 5))
      .resize(Array(4, 5), 4)
    val indices3 = Array(0, 1, 2, 3, 4, 4, 5)
    val indices4 = Array(0, 1, 2, 3, 3, 4, 2)
    val values2 = Array(6f, 7f, 8f, 9f, 10f, 11f, 12f)
    val input2 = Tensor.sparse(Array(indices3, indices4), values2, Array(6, 5))
      .narrow(1, 2, 4)
    val sparseModel = Sequential().add(ParallelTable().add(Identity()).add(Identity()))
      .add(SparseJoinTable(2))

    val sparseInput = T(input1, input2)
    val output = sparseModel.forward(sparseInput).toTensor[Float]

    val exceptedIndices1 = Array(0, 0, 1, 1, 2, 2, 3, 3, 3)
    val exceptedIndices2 = Array(0, 6, 1, 7, 2, 8, 3, 8, 9)
    val exceptedValues = Array(1f, 7f, 2, 8, 3, 9, 4, 10, 11)
    val exceptedOutput = Tensor.sparse(Array(exceptedIndices1, exceptedIndices2),
      exceptedValues, Array(4, 10))

    output should be (exceptedOutput)
  }

}

class SparseJoinTableSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val sparseJoinTable = SparseJoinTable[Float](2).setName("sparseJoinTable")
    val sparseModel = Sequential[Float]().
      add(ParallelTable[Float]().add(Identity[Float]()).add(Identity[Float]()))
      .add(sparseJoinTable)
    val input1 = Tensor[Float](4, 3).apply1(_ => Random.nextInt(2) * Random.nextFloat())
    val input2 = Tensor[Float](4, 2).apply1(_ => Random.nextInt(2) * Random.nextFloat())
    val sparseInput = T(Tensor.sparse(input1), Tensor.sparse(input2))
    runSerializationTest(sparseJoinTable, sparseInput)
  }
}
