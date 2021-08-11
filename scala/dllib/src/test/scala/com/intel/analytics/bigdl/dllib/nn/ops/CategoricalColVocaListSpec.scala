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

class CategoricalColVocaListSpec extends FlatSpec with Matchers{

  "CategoricalColVocaList operation with ignoring the outside values" should "work correctly" in {
    val input = Tensor[String](T(T("A"), T("B"), T("C"), T("A,D")))
    val indices = Array(Array(0, 1, 2, 3), Array(0, 0, 0, 0))
    val values = Array(0, 1, 2, 0)
    val shape = Array(4, 3)
    val expectOutput = Tensor.sparse(
      indices, values, shape
    )
    val output = CategoricalColVocaList[Double](
      vocaList = Array("A", "B", "C"),
      strDelimiter = ",",
      isSetDefault = false,
      numOovBuckets = 0
    ).forward(input)

    output should be(expectOutput)
  }

  "CategoricalColVocaList operation with default value" should "work correctly" in {
      val input = Tensor[String](T(T("A"), T("B"), T("C"), T("D")))
      val indices = Array(Array(0, 1, 2, 3), Array(0, 0, 0, 0))
      val values = Array(0, 1, 2, 3)
      val shape = Array(4, 4)
      val expectOutput = Tensor.sparse(
        indices, values, shape
      )
    val output = CategoricalColVocaList[Double](
      vocaList = Array("A", "B", "C"),
      strDelimiter = ",",
      isSetDefault = true,
      numOovBuckets = 0
    ).forward(input)

    output should be(expectOutput)
  }

  "CategoricalColVocaList operation with numOvvBucket" should "work correctly" in {
    val input = Tensor[String](T(T("A,B"), T("C"), T("B,C,D"), T("A,D")))
    val indices = Array(
      Array(0, 0, 1, 2, 2, 2, 3, 3),
      Array(0, 1, 0, 0, 1, 2, 0, 1))
    val values = Array(0, 1, 2, 1, 2, 4, 0, 4)
    val shape = Array(4, 5)
    val expectOutput = Tensor.sparse(
      indices, values, shape
    )
    val output = CategoricalColVocaList[Double](
      vocaList = Array("A", "B", "C"),
      strDelimiter = ",",
      numOovBuckets = 2
    ).forward(input)

    output should be(expectOutput)
  }

  "CategoricalColVocaList operation with 1-D input" should "work correctly" in {
    val input = Tensor[String](T("A", "B", "C", "D"))
    val indices = Array(Array(0, 1, 2, 3), Array(0, 0, 0, 0))
    val values = Array(0, 1, 2, 3)
    val shape = Array(4, 4)
    val expectOutput = Tensor.sparse(
      indices, values, shape
    )
    val output = CategoricalColVocaList[Double](
      vocaList = Array("A", "B", "C"),
      strDelimiter = ",",
      isSetDefault = true,
      numOovBuckets = 0
    ).forward(input)

    output should be(expectOutput)
  }
}

class CategoricalColVocaListSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val categoricalColVocaList = CategoricalColVocaList[Float](
      vocaList = Array("A", "B", "C"),
      strDelimiter = ",",
      isSetDefault = false,
      numOovBuckets = 0
    ).setName("categoricalColVocaList")
    val input = Tensor[String](T(T("A"), T("B"), T("C"), T("D")))
    runSerializationTest(categoricalColVocaList, input)
  }
}
