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

import com.intel.analytics.bigdl.nn.ops.BatchMatMul
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}


class GemmSpec extends FlatSpec with Matchers {

  "Gemm forward" should "work" in {
    val transA = false
    val transB = false

    val inputA = Tensor[Float](4, 2).rand()
    val inputB = Tensor[Float](2, 7).rand()
    val inputC = Tensor[Float](4, 7).rand()

    val tensorA = Input()
    val tensorB = Input()
    val tensorC = Input()

    val mul = BatchMatMul(adjX = transA, adjY = transB).inputs(Array(tensorA, tensorB))
    val add = CAddTable().inputs(Array(mul, tensorC))
    var model = Graph(Array(tensorA, tensorB, tensorC), add)

    var myGemm = new Gemm()
    val myInput = T(inputA, inputB, inputC)

    val out1 = model.forward(myInput)
    val out2 = myGemm.forward(myInput)

    out1 should be(out2)

  }


  "Gemm with transA forward" should "work" in {
    val transA = true
    val transB = false

    var inputA = Tensor[Float](2, 4).rand()
    var transInputA = inputA.t()
    var inputB = Tensor[Float](2, 7).rand()
    var transInputB = inputB.t()
    var inputC = Tensor[Float](4, 7).rand()

    val tensorA = Input()
    val tensorB = Input()
    val tensorC = Input()
    val mul = BatchMatMul(adjX = transA, adjY = transB).inputs(Array(tensorA, tensorB))
    val add = CAddTable().inputs(Array(mul, tensorC))
    var model = Graph(Array(tensorA, tensorB, tensorC), add)

    var myGemm = new Gemm(transA = true)

    val out1 = model.forward(T(inputA, inputB, inputC))
    val out2 = myGemm.forward(T(inputA, inputB, inputC))

    out1 should be(out2)

  }


  "Gemm with transB forward" should "work" in {
    val transA = false
    val transB = true

    var inputA = Tensor[Float](4, 2).rand()
    var transInputA = inputA.t()
    var inputB = Tensor[Float](7, 2).rand()
    var transInputB = inputB.t()
    var inputC = Tensor[Float](4, 7).rand()

    val tensorA = Input()
    val tensorB = Input()
    val tensorC = Input()
    val mul = BatchMatMul(adjX = transA, adjY = transB).inputs(Array(tensorA, tensorB))
    val add = CAddTable().inputs(Array(mul, tensorC))
    var model = Graph(Array(tensorA, tensorB, tensorC), add)

    var myGemm = new Gemm(transB = true)

    val out1 = model.forward(T(inputA, inputB, inputC))
    val out2 = myGemm.forward(T(inputA, inputB, inputC))

    out1 should be(out2)

  }


  "Gemm with transA & transB forward" should "work" in {
    val transA = true
    val transB = true

    var tensorA = Tensor[Float](2, 4).rand()
    var tensorB = Tensor[Float](7, 2).rand()
    var tensorC = Tensor[Float](4, 7).rand()

    val inputA = Input()
    val inputB = Input()
    val inputC = Input()

    val mul = BatchMatMul(adjX = transA, adjY = transB).inputs(Array(inputA, inputB))
    val add = CAddTable().inputs(Array(mul, inputC))
    var model = Graph(Array(inputA, inputB, inputC), add)

    var myGemm = new Gemm(transA = transA, transB = transB)

    val out1 = model.forward(T(tensorA, tensorB, tensorC))
    val out2 = myGemm.forward(T(tensorA, tensorB, tensorC))

    out1 should be(out2)

  }

}

class GemmSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val gemm = Gemm[Float]().setName("Gemm")

    val inputA = Tensor(2, 2).rand()
    val inputB = Tensor(2, 2).rand()
    val inputC = Tensor(2, 2).rand()
    val input = T(inputA, inputB, inputC)

    runSerializationTest(gemm, input)
  }
}
