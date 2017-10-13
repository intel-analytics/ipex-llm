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

}
