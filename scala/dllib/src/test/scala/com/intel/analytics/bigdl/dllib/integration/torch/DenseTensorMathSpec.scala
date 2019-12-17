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

package com.intel.analytics.bigdl.integration.torch

import com.intel.analytics.bigdl.tensor.Tensor

import scala.sys.process._

@com.intel.analytics.bigdl.tags.Serial
class DenseTensorMathSpec extends TorchSpec {

    "matrix + real" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val b = 100
    val code = "b = 100\noutcome = a + b"

    val start = System.nanoTime()
    val c = a + b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix + Real ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "matrix + matrix" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val b = Tensor[Double](200, 200).rand()
    val code = "outcome = a + b"

    val start = System.nanoTime()
    val c = a + b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a, "b" -> b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix + matrix ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9 + "\n")

    c should be(luaResult)
  }

  "matrix - real" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val b = 100
    val code = "b = 100\noutcome = a - b"

    val start = System.nanoTime()
    val c = a - b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix - Real ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "matrix - matrix" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val b = Tensor[Double](200, 200).rand()
    val code = "outcome = a - b"

    val start = System.nanoTime()
    val c = a - b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a, "b" -> b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix - matrix ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "negative matrix" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val code = "outcome = -a"

    val start = System.nanoTime()
    val c = -a
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== negative matrix ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "matrix / real" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val b = 100
    val code = "b = 100\noutcome = a / b"

    val start = System.nanoTime()
    val c = a / b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix / Real ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "matrix ./ matrix" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val b = Tensor[Double](200, 200).rand()
    val code = "outcome = a:cdiv(b)"

    val start = System.nanoTime()
    val c = a / b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a, "b" -> b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix ./ matrix ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "matrix * real" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val b = 100
    val code = "b = 100\noutcome = a * b"

    val start = System.nanoTime()
    val c = a * b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix * real ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }


  "matrix * matrix" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val b = Tensor[Double](200, 200).rand()
    val code = "outcome = a * b"

    val start = System.nanoTime()
    val c = a * b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a, "b" -> b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix * matrix ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "matrix sumAll" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val code = "outcome = a:sum()"

    val start = System.nanoTime()
    val c = a.sum()
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Double]

    println("\n======== matrix sumAll ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "matrix sumDimension" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val code = "outcome = a:sum(1)"

    val start = System.nanoTime()
    val c = a.sum(1)
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix sumDimension ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "matrix max" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val code = "outcome = a:max()"

    val start = System.nanoTime()
    val c = a.max()
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Double]

    println("\n======== matrix max ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "matrix conv2" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val b = Tensor[Double](200, 200).rand()
    val code = "outcome = torch.conv2(a, b, 'V')"

    val start = System.nanoTime()
    val c = a.conv2(b, 'V')
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a, "b" -> b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix conv2 ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "matrix xcorr2" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val b = Tensor[Double](200, 200).rand()
    val code = "outcome = torch.xcorr2(a, b, 'V')"

    val start = System.nanoTime()
    val c = a.xcorr2(b, 'V')
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a, "b" -> b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix xcorr2 ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "matrix add" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val b = Tensor[Double](200, 200).rand()
    val m = 20
    val code = "m = 20\noutcome = a:add(m, b)"

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a, "b" -> b), Array("outcome"))
    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    val c = a.add(m, b)
    val end = System.nanoTime()
    val scalaTime = end - start

    println("\n======== matrix add ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "matrix cmul" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val b = Tensor[Double](200, 200).rand()
    val code = "outcome = a:cmul(b)"

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a, "b" -> b), Array("outcome"))
    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    val c = a.cmul(b)
    val end = System.nanoTime()
    val scalaTime = end - start

    println("\n======== matrix cmul ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "matrix mul" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val b = Tensor[Double](200, 200).rand()
    val m = 20
    val code = "m = 20\noutcome = a:mul(b, m)"

    val start = System.nanoTime()
    val c = a.mul(b, m)
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a, "b" -> b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix mul ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "matrix div" should "return correct value" in {
    torchCheck()
    val a = Tensor[Double](200, 200).rand()
    val b = 20
    val code = "b = 20\noutcome = a:div(b)"

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a), Array("outcome"))
    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    val c = a.div(b)
    val end = System.nanoTime()
    val scalaTime = end - start


    println("\n======== matrix div ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c should be(luaResult)
  }

  "addmm" should "return correct value" in {
    torchCheck()

    val a = Tensor[Double](200, 200).rand()
    val b = Tensor[Double](200, 200).rand()

    val start = System.nanoTime()
    val c = Tensor[Double]().resize(Array(a.size(1), b.size(2))).addmm(a, b)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "outcome = torch.Tensor(a:size(1), b:size(2)):fill(0)\noutcome:addmm(a, b)"

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a, "b" -> b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== addmm ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c.map(luaResult.asInstanceOf[Tensor[Double]], (tmp1, tmp2) => {
      tmp1 should be(tmp2 +- 1e-6)
      tmp1
    })
  }

  "addr" should "return correct value" in {
    torchCheck()

    val a = Tensor[Double](200).rand()
    val b = Tensor[Double](200).rand()

    val start = System.nanoTime()
    val c = Tensor[Double]().resize(Array(a.size(1), b.size(1))).addr(a, b)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "outcome = torch.Tensor(a:size(1), b:size(1)):fill(0)\noutcome:addr(a, b)"

    val (luaTime, torchResult) = TH.run(code, Map("a" -> a, "b" -> b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== addr ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime / 1e9.toDouble + "\n")

    c.map(luaResult.asInstanceOf[Tensor[Double]], (tmp1, tmp2) => {
      tmp1 should be(tmp2 +- 1e-6)
      tmp1
    })
  }

}


