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

import scala.util.Random

class GemmSpec extends FlatSpec with Matchers {

  "Gemm forward" should "work" in {

    val inputA = Tensor[Float](2, 2)
    val inputB = Tensor[Float](2, 2)
    val inputC = Tensor[Float](2, 2)

    inputA.setValue(1, 1, 1)
    inputA.setValue(1, 2, 2)
    inputA.setValue(2, 1, 3)
    inputA.setValue(2, 2, 4)

    inputB.setValue(1, 1, 1)
    inputB.setValue(1, 2, 2)
    inputB.setValue(2, 1, 3)
    inputB.setValue(2, 2, 4)

    inputC.setValue(1, 1, 1)
    inputC.setValue(1, 2, 2)
    inputC.setValue(2, 1, 3)
    inputC.setValue(2, 2, 4)

    val tensorA = Input()
    val tensorB = Input()
    val tensorC = Input()
    val mul = BatchMatMul().inputs(Array(tensorA, tensorB))
    val add = CAddTable().inputs(Array(mul, tensorC))
    var model = Graph(Array(tensorA, tensorB, tensorC), add)

    var myGemm = new Gemm()


    val myInput = T(inputA, inputB, inputC)

    val out1 = model.forward(myInput)
    val out2 = myGemm.forward(myInput)

    out1 should be(out2)
  }

}

class GemmSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val gemm = Gemm[Float](alpha = 1, beta = 1).setName("Gemm")

    val inputA = Tensor(2, 2)
    val inputB = Tensor(2, 2)
    val inputC = Tensor(2, 2)

    inputA.setValue(1, 1, 1)
    inputA.setValue(1, 2, 2)
    inputA.setValue(2, 1, 3)
    inputA.setValue(2, 2, 4)

    inputB.setValue(1, 1, 1)
    inputB.setValue(1, 2, 2)
    inputB.setValue(2, 1, 3)
    inputB.setValue(2, 2, 4)

    inputC.setValue(1, 1, 1)
    inputC.setValue(1, 2, 2)
    inputC.setValue(2, 1, 3)
    inputC.setValue(2, 2, 4)

    val input = T(inputA, inputB, inputC)

    runSerializationTest(gemm, input)
  }
}