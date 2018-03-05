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

import com.intel.analytics.bigdl.tensor.{DenseType, SparseType, Tensor}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class Kv2TensorSpec extends FlatSpec with Matchers {

  protected def randDoubles(length: Int,
                            lp: Double = 0.0,
                            up: Double = 1.0): Array[Double] = {
    (1 to length).map(_ => lp + (up - lp) * Random.nextDouble()).toArray
  }

  protected def randKVMap(size: Int,
                          numActive: Int,
                          lp: Double = 0.0,
                          up: Double = 1.0): Map[Int, Double] = {
    require(numActive <= size)
    val keys = Random.shuffle((0 until size).toList).take(numActive)
    val values = randDoubles(numActive, lp, up)
    keys.zip(values).toMap
  }
  val batchLen = 3
  val numActive = Array(2, 3, 5)
  val feaLen = 8
  val originData = new ArrayBuffer[String]()
  val originArr = new ArrayBuffer[Table]()
  val indices0 = new ArrayBuffer[Int]()
  val indices1 = new ArrayBuffer[Int]()
  val values = new ArrayBuffer[Double]()
  for (i <- 0 until batchLen) {
    val kvMap = randKVMap(feaLen, numActive(i))
    val kvStr = kvMap.map(data => s"${data._1}:${data._2}").mkString(",")
    originData += kvStr
    originArr += T(kvStr)
    indices0 ++= ArrayBuffer.fill(numActive(i))(i)
    val kvArr = kvMap.toArray
    indices1 ++= kvArr.map(kv => kv._1)
    values ++= kvArr.map(kv => kv._2)
  }
  val originTable = T.array(originArr.toArray)
  val indices = Array(indices0.toArray, indices1.toArray)
  val shape = Array(batchLen, feaLen)

  "Kv2Tensor operation kvString to SparseTensor" should "work correctly" in {
    val input =
      T(
        Tensor[String](originTable),
        Tensor[Int](Array(feaLen), shape = Array[Int]())
      )

    val expectOutput =
      Tensor.sparse[Double](
        indices = indices,
        values = values.toArray,
        shape = shape
      )
    val output = Kv2Tensor[Double, Double](transType = 1)
      .forward(input)

    output should be(expectOutput)
  }

  "Kv2Tensor operation kvString to DenseTensor" should "work correctly" in {
    val input =
      T(
        Tensor[String](originTable),
        Tensor[Int](Array(feaLen), shape = Array[Int]())
      )

    val expectOutput =
      Tensor.dense(Tensor.sparse[Double](
        indices = indices,
        values = values.toArray,
        shape = shape
      ))
    val output = Kv2Tensor[Double, Double](transType = 0)
      .forward(input)

    output should be(expectOutput)
  }
}

class Kv2TensorSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val kv2tensor = Kv2Tensor[Float, Float](
      kvDelimiter = ",", itemDelimiter = ":", transType = 0
    ).setName("kv2tensor")
    val input = T(
      Tensor[String](
        T(T("0:0.1,1:0.2"), T("1:0.3,3:0.5"), T("2:0.15,4:0.25"))),
      Tensor[Int](Array(5), shape = Array[Int]())
    )
    runSerializationTest(kv2tensor, input)
  }
}
