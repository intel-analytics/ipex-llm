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

import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{SparseTensor, Tensor}
import com.intel.analytics.bigdl.utils.{RandomGenerator, T}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class SparseLinearSpec extends FlatSpec with Matchers with BeforeAndAfter {

  before {
    RandomGenerator.RNG.setSeed(100)
  }

  "Sparse Linear" should "return the same result with Linear" in {
    val weight = Tensor.range(1, 8, 1).resize(2, 4)
    val bias = Tensor(2)
    val gradOutput = Tensor.range(1, 4, 1).resize(2, 2)
    val sl = SparseLinear(4, 2)
    val l = Linear(4, 2)
    l.weight.copy(weight)
    l.bias.copy(bias)
    sl.weight.copy(weight)
    sl.bias.copy(bias)
    val input = Tensor(2, 4)
    input.setValue(1, 1, 1f)
    input.setValue(2, 3, 3f)
    val sparseInput = Tensor.sparse(input)
    val out1 = sl.forward(sparseInput)
    sl.backward(sparseInput, gradOutput)
    val out2 = l.forward(input)
    l.backward(input, gradOutput)
    out1 should be (out2)
    sl.getParameters()._2 should be (l.getParameters()._2)
  }

  "Sparse Linear" should "return the same result with Linear 2" in {
    val gradOutput = Tensor(2, 2).rand()
    val input = Tensor(2, 4).rand()
    val sl = SparseLinear(4, 2)
    val l = Linear(4, 2)
    l.weight.copy(sl.weight)
    l.bias.copy(sl.bias)
    val sparseInput = Tensor.sparse(input)
    val out1 = sl.forward(sparseInput)
    sl.backward(sparseInput, gradOutput)
    val out2 = l.forward(input)
    l.backward(input, gradOutput)
    out1 should be (out2)
    sl.getParameters()._2 should be (l.getParameters()._2)
  }


  "Sparse Linear" should "return the same result with Linear 3" in {
    val gradOutput = Tensor(2, 2).rand()
    val input = Tensor(2, 4).rand()
    val sl = SparseLinear(4, 2, backwardStart = 1, backwardLength = 4)
    val l = Linear(4, 2)
    l.weight.copy(sl.weight)
    l.bias.copy(sl.bias)
    val sparseInput = Tensor.sparse(input)
    val out1 = sl.forward(sparseInput)
    val gradInput1 = sl.backward(sparseInput, gradOutput)
    val out2 = l.forward(input)
    val gradInput2 = l.backward(input, gradOutput)
    out1 should be (out2)
    gradInput1 should be (gradInput2)
    sl.getParameters()._2 should be (l.getParameters()._2)
  }

  "Sparse Linear" should "return the same result with Linear 4" in {
    val gradOutput = Tensor(3, 2).rand()
    val input = Tensor(3, 4).rand()
    val sl = SparseLinear(4, 2, backwardStart = 1, backwardLength = 4)
    val l = Linear(4, 2)
    l.weight.copy(sl.weight)
    l.bias.copy(sl.bias)
    val sparseInput = Tensor.sparse(input)
    val out1 = sl.forward(sparseInput)
    val gradInput1 = sl.backward(sparseInput, gradOutput)
    val out2 = l.forward(input)
    val gradInput2 = l.backward(input, gradOutput)
    out1 should be (out2)
    gradInput1 should be (gradInput2)
    sl.getParameters()._2 should be (l.getParameters()._2)
  }

  "Sparse Linear" should "return the same result with Linear 5" in {
    val gradOutput = Tensor(4, 2).rand()
    val input = Tensor(4, 10).apply1(_ => Random.nextInt(10) / 9 * Random.nextFloat())
    val sl = SparseLinear(10, 2, backwardStart = 5, backwardLength = 5)
    val l = Linear(10, 2)
    l.weight.copy(sl.weight)
    l.bias.copy(sl.bias)
    val sparseInput = Tensor.sparse(input)
    val out1 = sl.forward(sparseInput)
    val gradInput1 = sl.backward(sparseInput, gradOutput)
    val out2 = l.forward(input)
    val gradInput2 = l.backward(input, gradOutput)
    out1 should be (out2)
    gradInput1 should be (gradInput2.narrow(2, 5, 5))
    sl.getParameters()._2 should be (l.getParameters()._2)
  }

  "Sparse Linear" should "return the same result with Linear 6" in {
    val gradOutput = Tensor(4, 2).rand()
    val input = Tensor(4, 3).apply1(_ => Random.nextInt(5) / 4 * Random.nextFloat())
    val input2 = Tensor(4, 2).apply1(_ => Random.nextInt(2) * Random.nextFloat())
    val sl = SparseLinear(5, 2, backwardStart = 1, backwardLength = 5)
    val sparseModel = Sequential().add(ParallelTable().add(Identity()).add(Identity()))
      .add(SparseJoinTable(2))
      .add(sl)
    val l = Linear(5, 2)
    l.weight.copy(sl.weight)
    l.bias.copy(sl.bias)
    val denseInput = Tensor(4, 5)
    denseInput.narrow(2, 1, 3).copy(input)
    denseInput.narrow(2, 4, 2).copy(input2)

    val sparseInput = T(Tensor.sparse(input), Tensor.sparse(input2))
    Tensor.sparse(denseInput)
    val out1 = sparseModel.forward(sparseInput).toTensor[Float]
    val gradInput1 = sparseModel.backward(sparseInput, gradOutput)

    val out2 = l.forward(denseInput)
    val gradInput2 = l.backward(denseInput, gradOutput)
    out1 shouldEqual out2
    sl.gradInput should be (gradInput2)
    sl.getParameters()._2 should be (l.getParameters()._2)
  }

  "Sparse Linear" should "return the same result with Linear 7" in {
    RandomGenerator.RNG.setSeed(10)
    val rnd = new Random(10)
    val gradOutput = Tensor(4, 2).rand()
    val input = Tensor(4, 1023213).apply1(_ => rnd.nextInt(100000) / 99999 * rnd.nextFloat())
    val input2 = Tensor(4, 50).apply1(_ => rnd.nextInt(2) * rnd.nextFloat())
    val sl = SparseLinear(1023263, 2, backwardStart = 1, backwardLength = 1023263)
    val sj = SparseJoinTable(2)
    val sparseModel = Sequential().add(ParallelTable().add(Identity()).add(Identity()))
      .add(sj)
      .add(sl)
    val l = Linear(1023263, 2)
    l.weight.copy(sl.weight)
    l.bias.copy(sl.bias)
    val denseInput = Tensor(4, 1023263)
    denseInput.narrow(2, 1, 1023213).copy(input)
    denseInput.narrow(2, 1023214, 50).copy(input2)

    val sparseInput = T(Tensor.sparse(input), Tensor.sparse(input2))
    val si = Tensor.sparse(denseInput)
    val aaa = sl.forward(si).toTensor[Float].clone()
    val out1 = sparseModel.forward(sparseInput).toTensor[Float]
    val gradInput1 = sparseModel.backward(sparseInput, gradOutput)
//
    val out2 = l.forward(denseInput)
    val gradInput2 = l.backward(denseInput, gradOutput)
    aaa shouldEqual out2
    sj.output shouldEqual si
    out1 shouldEqual out2
    sl.gradInput should be (gradInput2)
    sl.getParameters()._2.equals(l.getParameters()._2) shouldEqual true
  }
}

class SparseLinearSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val sparseLinear = SparseLinear[Float](4, 2).setName("sparseLinear")
    val input = Tensor[Float](2, 4).apply1(_ => Random.nextFloat())
    val sparseInput = Tensor.sparse(input)
    runSerializationTest(sparseLinear, sparseInput)
  }
}
