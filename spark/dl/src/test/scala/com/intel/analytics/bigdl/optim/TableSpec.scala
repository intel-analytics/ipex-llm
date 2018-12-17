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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.nn.{ConcatTable, Linear}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Properties

@com.intel.analytics.bigdl.tags.Parallel
class TableSpec extends FlatSpec with Matchers {
  "hashcode()" should "behave correctly" in {
    val input1 = Tensor[Double](3, 3).randn()
    val input2 = Tensor[Double](3, 3).randn()

    val t1 = T()
    val t2 = T()
    val t3 = T(1 -> input1, 2 -> input2)
    val t4 = T(1 -> input1, 2 -> input2)
    val t5 = T(1 -> input1)
    val tNull = null
    val tensor = Tensor[Double]()

    t1.hashCode() should equal (t2.hashCode().hashCode())
    t1.hashCode() should not equal t3.hashCode()
    t3.hashCode() should equal (t4.hashCode())
    t4.hashCode() should not equal t5.hashCode()
  }

  "Test type of Activity" should "be correct" in {
    val t: Table = T()
    t.isTensor should be(false)
    t.isTable should be(true)
  }

  "equals()" should "behave correctly" in {
    val input1 = Tensor[Double](3, 3).randn()
    val input2 = Tensor[Double](3, 3).randn()

    val t1 = T()
    val t2 = T()
    val t3 = T(1 -> input1, 2 -> input2)
    val t4 = T(1 -> input1, 2 -> input2)
    val t5 = T(1 -> input1)
    val tNull = null
    val tensor = Tensor[Double]()

    t1 should equal (t2)
    t1 should not equal t3
    t3 should equal (t4)
    t4 should not equal t5
    t4 should not equal tNull
    t4 should not equal tensor
  }

  "clone" should "work correctly" in {
    val state1 = T()
    state1("a") = "1"
    state1("b") = 1

    val state2 = state1.clone()
    state2.get[String]("a").get should be("1")
    state2.get[Int]("b").get should be(1)
  }

  "toString" should "return correct value" in {
    val state = T()
    state(1) = Tensor[Double](3, 3).zero()
    state(2) = T(1 -> "b", 2 -> T(1 -> "d"))
    print(state.toString)
    // Handle different behavior of toString in scala 2.10 and 2.11
    if(Properties.versionNumberString.contains("2.10")) {
      state.toString() should be(" {\n\t2:  {\n\t   " +
        "\t2:  {\n\t   \t   \t1: d\n\t   \t    }\n\t   \t1: b\n\t    }" +
        "\n\t1: 0.0\t0.0\t0.0\t\n\t   0.0\t0.0\t0.0\t\n\t   0.0\t0.0\t0.0\t\n\t   " +
        "[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x3]\n }")
    } else if (Properties.versionNumberString.contains("2.11")) {
      state.toString() should be(" {\n\t2:  {\n\t   " +
        "\t2:  {\n\t   \t   \t1: d\n\t   \t    }\n\t   \t1: b\n\t    }" +
        "\n\t1: 0.0\t0.0\t0.0\t\n\t   0.0\t0.0\t0.0\t\n\t   0.0\t0.0\t0.0\t\n\t   " +
        "[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]\n }")
    }
  }

  "init with data" should "return correct value" in {
    val state = T(1, 2, 3)

    println(state)
    state[Int](1) should be(1)
    state[Int](2) should be(2)
    state[Int](3) should be(3)
  }

  "init with tuples" should "return correct value" in {
    val state = T("a" -> "b", 2 -> "d", "e" -> 1)

    println(state)
    state[String]("a") should be("b")
    state[String](2) should be("d")
    state[Int]("e") should be(1)
  }

  "copy from another table" should "return correct value" in {
    val s1 = T("a" -> 1, "b" -> 2)
    val s2 = T("b" -> 3, "c" -> 4)
    s1.add(s2)

    s1[Int]("a") should be(1)
    s1[Int]("b") should be(3)
    s1[Int]("c") should be(4)
  }

  "insert to table" should "return correct value" in {
    val t = T(1, 2, 3)

    t.insert(4)

    println(t)
    t[Int](1) should be(1)
    t[Int](2) should be(2)
    t[Int](3) should be(3)
    t[Int](4) should be(4)

    t.insert(6, 6)
    println(t)
    t[Int](1) should be(1)
    t[Int](2) should be(2)
    t[Int](3) should be(3)
    t[Int](4) should be(4)
    t[Int](6) should be(6)

    t.insert(5, 5)
    println(t)
    t[Int](1) should be(1)
    t[Int](2) should be(2)
    t[Int](3) should be(3)
    t[Int](4) should be(4)
    t[Int](5) should be(5)
    t[Int](6) should be(6)


    t.insert(1, 0)
    println(t)
    t[Int](1) should be(0)
    t[Int](2) should be(1)
    t[Int](3) should be(2)
    t[Int](4) should be(3)
    t[Int](5) should be(4)
    t[Int](6) should be(5)
    t[Int](7) should be(6)

    t(10) = 10
    t.insert(6, 8)
    t[Int](1) should be(0)
    t[Int](2) should be(1)
    t[Int](3) should be(2)
    t[Int](4) should be(3)
    t[Int](5) should be(4)
    t[Int](6) should be(8)
    t[Int](7) should be(5)
    t[Int](8) should be(6)
    t[Int](10) should be(10)

    t(9) = 9
    t.insert(6, 8.1)
    t[Int](1) should be(0)
    t[Int](2) should be(1)
    t[Int](3) should be(2)
    t[Int](4) should be(3)
    t[Int](5) should be(4)
    t[Int](6) should be(8.1)
    t[Int](7) should be(8)
    t[Int](8) should be(5)
    t[Int](9) should be(6)
    t[Int](10) should be(9)
    t[Int](11) should be(10)
  }

  "table size" should "return correct value" in {
    val t = T(1, 2, 3, 4, 5, 6)
    t.length() should be(6)
  }

  "remove from table" should "return correct value" in {
    val t = T(1, 2, 3, 4, 5, 6)

    var r = t.remove[Int]()
    r.get should be(6)
    t[Int](1) should be(1)
    t[Int](2) should be(2)
    t[Int](3) should be(3)
    t[Int](4) should be(4)
    t[Int](5) should be(5)


    r = t.remove[Int](10)
    r.isDefined should be(false)

    t.insert(11, 11)
    r = t.remove[Int](11)
    r.get should be(11)
    t[Int](1) should be(1)
    t[Int](2) should be(2)
    t[Int](3) should be(3)
    t[Int](4) should be(4)
    t[Int](5) should be(5)


    r = t.remove[Int](2)
    r.get should be(2)
    t[Int](1) should be(1)
    t[Int](2) should be(3)
    t[Int](3) should be(4)
    t[Int](4) should be(5)

    t.remove()
    t.remove()
    t.remove()
    t.remove()
    t.length() should be(0)
    r = t.remove()
    r.isDefined should be(false)
    t.length() should be(0)
    t.insert(100)
    t[Int](1) should be(100)
  }

  "clear from table" should "return correct value" in {
    val t = T(1, 2, 3, 4, 5, 6)

    val r = t.clear()
    r.get(1) should be(None)
    r.contains(1) should be(false)
    r.length() should be(0)
    t.length() should be(0)

    t.insert(1, 1)
    t.length() should be(1)

    t.insert(100)
    t[Int](2) should be(100)
  }

  "toTensor" should "work correclty" in {
    val input = Tensor[Float](3, 4)
    val module = ConcatTable[Tensor[Float], Float]()
      .add(Linear(4, 5))
      .add(Linear(4, 5))
    val output = module.forward(input)

    try {
      output.toTensor[Float]
      fail()
    } catch {
      case ex: IllegalArgumentException =>
    }
  }

  "toTable" should "work correclty" in {
    val input = Tensor[Float](3, 4)
    val module = ConcatTable[Tensor[Float], Float]()
      .add(Linear(4, 5))
      .add(Linear(4, 5))
    val output = module.forward(input)

    output.toTable should be(output)
  }

  "toSeq" should "work correclty" in {

    val t = T(Tensor[Double](T(1.0)), Tensor[Double](T(2.0)))

    t.toSeq[Tensor[Double]] should be (Seq(Tensor[Double](T(1.0)), Tensor[Double](T(2.0))))
  }
}
