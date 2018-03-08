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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class PredictionServiceSpec extends FlatSpec with Matchers {

  // sharing weights for testModule and testModule2
  private val linearWeights = Tensor[Float](5, 10).rand()
  private val linearBias = Tensor[Float](5).rand()
  private val linear2Weights = Tensor[Float](1, 5).rand()
  private val linear2Bias = Tensor[Float](1).rand()

  private val testModule = {
    val input = Input[Float]()
    val linear = Linear[Float](10, 5,
      initWeight = linearWeights, initBias = linearBias).inputs(input)
    val relu = ReLU[Float]().inputs(linear)
    val linear2 = Linear[Float](5, 1,
      initWeight = linear2Weights, initBias = linear2Bias).inputs(relu)
    val sigmoid = Sigmoid[Float]().inputs(linear2)
    Graph[Float](input, sigmoid)
  }

  private val testModule2 = {
    val (input1, input2) = (Input[Float](), Input[Float]())
    val concat = JoinTable[Float](1, 1).inputs(input1, input2)
    val linear = Linear[Float](10, 5,
      initWeight = linearWeights, initBias = linearBias).inputs(concat)
    val relu = ReLU[Float]().inputs(linear)
    val linear2 = Linear[Float](5, 1,
      initWeight = linear2Weights, initBias = linear2Bias).inputs(relu)
    val sigmoid = Sigmoid[Float]().inputs(linear2)
    Graph[Float](Array(input1, input2), sigmoid)
  }

  "Tensor/ByteArray convert" should "work properly" in {
    testTensorSerialize(0)
    testTensorSerialize(0L)
    testTensorSerialize(0.0f)
    testTensorSerialize(0.0)
    testTensorSerialize(true)
    testTensorSerialize('a')
    testTensorSerialize("aa")
  }

  private val testTensorSerialize = (flag: Any) => {
    val tensor = flag match {
      case _: Int => Tensor[Int](2, 3).randn()
      case _: Long => Tensor[Long](2, 3).randn()
      case _: Float => Tensor[Float](2, 3).randn()
      case _: Double => Tensor[Double](2, 3).randn()
      case _: Boolean => Tensor[Boolean](T(true, false, T(true, false)))
      case _: String => Tensor[String](T("a", T("b", "c"), T("d", "e")))
      case _: Char => Tensor[Char](T('a', T('b', 'c', 'd')))
    }
    val bytes = PredictionService.serializeActivity(tensor)
    val tensor2 = PredictionService.deSerializeActivity(bytes)
    tensor shouldEqual tensor2
  }

  "Table/ByteArray convert" should "work properly" in {
    // test with increment keys
    var table = T.seq((1 to 5).map(_ => Tensor[Double](3, 5).randn()))
    var bytes = PredictionService.serializeActivity(table)
    var table2 = PredictionService.deSerializeActivity(bytes)
    table shouldEqual table2

    // test with double keys
    table = T()
    (1 to 5).foreach(_ =>
      table.update(Random.nextDouble(), Tensor[Double](3, 5).randn()))
    bytes = PredictionService.serializeActivity(table)
    table2 = PredictionService.deSerializeActivity(bytes)
    table shouldEqual table2

    // test with Tensor[String] keys
    table = T()
    (1 to 5).foreach(i =>
      table.update(Tensor.scalar(i.toString), Tensor[Double](3, 5).randn()))
    bytes = PredictionService.serializeActivity(table)
    table2 = PredictionService.deSerializeActivity(bytes)
    table shouldEqual table2
  }

  "PredictionService.predict" should "return a error message when exception caught" in {
    // forward exception
    val service = PredictionService[Float](testModule, 2)
    val invalidTensor = Tensor[Float](2, 11).randn()
    var eTensor = service.predict(invalidTensor).asInstanceOf[Tensor[String]]
    eTensor.isScalar shouldEqual true
    eTensor.value().contains("running forward") shouldEqual true

    // Clone Result
    val module = ParallelTable[Float]()
      .add(SplitTable[Float](1, 1)).add(SplitTable[Float](1, 1))
    val tableInput = T(Tensor[Float](2, 3).randn(), Tensor[Float](3, 2).randn())
    eTensor = PredictionService(module, 2).predict(tableInput).asInstanceOf[Tensor[String]]
    eTensor.isScalar shouldEqual true
    eTensor.value().contains("Clone Result") shouldEqual true

    // DeSerialize exception
    val tensor = Tensor[Float](2, 10).randn()
    val bytes = PredictionService.serializeActivity(tensor)
    val invalidBytes = bytes.map(e => (e + 1).toByte)
    val eBytesOut = service.predict(invalidBytes)
    eTensor = PredictionService.deSerializeActivity(eBytesOut)
      .asInstanceOf[Tensor[String]]
    eTensor.isScalar shouldEqual true
    eTensor.value().contains("DeSerialize Input") shouldEqual true
  }

  "PredictionService" should "work properly with concurrent calls" in {
    val service = PredictionService[Float](testModule, 4)
    val service2 = PredictionService[Float](testModule, 1)

    val sumResults = (1 to 100).par.map { _ =>
      val tensor = Tensor[Float](2, 10).randn()
      val output = service.predict(tensor).asInstanceOf[Tensor[Float]]
      val output2 = service2.predict(tensor).asInstanceOf[Tensor[Float]]
      output.size() shouldEqual Array(2, 1)
      output shouldEqual output2
      output.squeeze().toArray().sum
    }
    // Check whether instances have independent status(outputs of each Layer).
    sumResults.toList.distinct.lengthCompare(90) > 0 shouldEqual true
  }

  "PredictionService" should "work properly with byteArray data" in {
    var service = PredictionService[Float](testModule, 2)
    val tensor = Tensor[Float](2, 10).randn()
    val input = PredictionService.serializeActivity(tensor)
    val output = PredictionService.deSerializeActivity(service.predict(input))
      .asInstanceOf[Tensor[Float]]
    output.size() shouldEqual Array(2, 1)

    service = PredictionService[Float](testModule2, 2)
    val input2 = PredictionService.serializeActivity(
      T(tensor.narrow(2, 1, 6), tensor.narrow(2, 7, 4)))
    val output2 = PredictionService.deSerializeActivity(service.predict(input2))
      .asInstanceOf[Tensor[Float]]
    // TestModule and testModule2 have same network weights/bias and same inputs,
    // so their outputs should be equal.
    output shouldEqual output2
  }

}
