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
package com.intel.analytics.bigdl.models.utils

import java.nio.ByteOrder

import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn.tf.Const
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.commons.lang3.SerializationUtils
import com.intel.analytics.bigdl.utils.SparkContextLifeCycle
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ModelBroadcastSpec extends SparkContextLifeCycle with Matchers {

  override def appName: String = "ModelBroadcast"

  override def afterTest: Any = {
    System.clearProperty("bigdl.ModelBroadcastFactory")
  }

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  "model broadcast" should "work properly" in {
    val model = LeNet5(10)

    val modelBroadCast = ModelBroadcast[Float]().broadcast(sc, model)
    modelBroadCast.value().toString should be(model.toString)
    modelBroadCast.value().parameters()._1 should be(model.parameters()._1)
  }

  "model broadcast with const" should "forward properly" in {
    val input1 = Identity[Float]().inputs()
    val input2 = Const[Float, Float](Tensor[Float].range(1, 6, 1)).setName("const").inputs()
    val output = CAddTable[Float]().inputs(input1, input2)
    val model = Graph(input1, output)
    System.setProperty("bigdl.ModelBroadcastFactory",
      "com.intel.analytics.bigdl.models.utils.ProtoBufferModelBroadcastFactory")
    val modelBroadCast = ModelBroadcast[Float]().broadcast(sc, model)

    val testModel = modelBroadCast.value()
    val testInput = Tensor[Float].range(2, 7, 1)
    val testOutput = testModel.forward(testInput)
    testOutput should be (Tensor[Float].range(3, 13, 2))
  }

  "model broadcast with const" should "const shared properly" in {
    val input1 = Identity[Float]().inputs()
    val input2 = Const[Float, Float](Tensor[Float].range(1, 6, 1)).setName("const").inputs()
    val output = CAddTable[Float]().inputs(input1, input2)
    val model = Graph(input1, output)

    val modelBroadCast = ModelBroadcast[Float]().broadcast(sc, model)
    val model1 = modelBroadCast.value().asInstanceOf[Graph[Float]]
    val model2 = modelBroadCast.value().asInstanceOf[Graph[Float]]
    val const1 = model1.findModules("Const")(0).asInstanceOf[Const[Float, Float]]
    val const2 = model2.findModules("Const")(0).asInstanceOf[Const[Float, Float]]
    const1.value should be (const2.value)
    const1.value.storage() should be (const2.value.storage())
  }

  "model broadcast with const" should "const shared properly 2" in {
    val input1 = Identity[Float]().inputs()
    val input2 = Const[Float, Float](Tensor[Float].range(1, 6, 1)).setName("const").inputs()
    val output = CAddTable[Float]().inputs(input1, input2)
    val model = Sequential[Float]().add(Graph(input1, output))

    val modelBroadCast = ModelBroadcast[Float]().broadcast(sc, model)
    val model1 = modelBroadCast.value().asInstanceOf[Sequential[Float]]
    val model2 = modelBroadCast.value().asInstanceOf[Sequential[Float]]
    val const1 = model1.findModules("Const")(0).asInstanceOf[Const[Float, Float]]
    val const2 = model2.findModules("Const")(0).asInstanceOf[Const[Float, Float]]
    const1.value should be (const2.value)
    const1.value.storage() should be (const2.value.storage())
  }

  "model broadcast with applyProtoBuffer" should "work properly" in {
    val model = LeNet5(10)

    System.setProperty("bigdl.ModelBroadcastFactory",
      "com.intel.analytics.bigdl.models.utils.ProtoBufferModelBroadcastFactory")
    val modelBroadCast = ModelBroadcast[Float]().broadcast(sc, model)
    modelBroadCast.value().toString should be(model.toString)
    modelBroadCast.value().parameters()._1 should be(model.parameters()._1)
  }

  "model broadcast with getParameters" should "work properly" in {
    val model = LeNet5(10)
    model.getParameters()

    val modelBroadCast = ModelBroadcast[Float]().broadcast(sc, model)
    modelBroadCast.value().toString should be(model.toString)
    modelBroadCast.value().parameters()._1 should be(model.parameters()._1)
  }

  "model broadcast with applyProtoBuffer with getParameters" should "work properly" in {
    val model = LeNet5(10)
    model.getParameters()


    System.setProperty("bigdl.ModelBroadcastFactory",
      "com.intel.analytics.bigdl.models.utils.ProtoBufferModelBroadcastFactory")
    val modelBroadCast = ModelBroadcast[Float]().broadcast(sc, model)
    modelBroadCast.value().toString should be(model.toString)
    modelBroadCast.value().parameters()._1 should be(model.parameters()._1)
  }

  "quantized model broadcast" should "work properly" in {
    val model = LeNet5(10).quantize()

    val modelBroadCast = ModelBroadcast[Float]().broadcast(sc, model)
    modelBroadCast.value().toString should be(model.toString)
    modelBroadCast.value().parameters()._1 should be(model.parameters()._1)
  }

  "quantized model broadcast with applyProtoBuffer" should "work properly" in {
    val model = LeNet5(10).quantize()

    System.setProperty("bigdl.ModelBroadcastFactory",
      "com.intel.analytics.bigdl.models.utils.ProtoBufferModelBroadcastFactory")
    val modelBroadCast = ModelBroadcast[Float]().broadcast(sc, model)
    modelBroadCast.value().toString should be(model.toString)
    modelBroadCast.value().parameters()._1 should be(model.parameters()._1)
  }

  "quantized multi groups model" should "work properly" in {
    val model = Sequential[Float]()
      .add(SpatialConvolution[Float](2, 4, 4, 4, 1, 1, 0, 0, 2))
      .quantize()

    val modelBroadCast = ModelBroadcast[Float]().broadcast(sc, model)
    modelBroadCast.value().toString should be(model.toString)
    modelBroadCast.value().parameters()._1 should be(model.parameters()._1)
  }

  "model info serialized" should "not be null" in {
    val model = LeNet5(10).cloneModule()
    val info = ModelInfo[Float]("124339", model)

    val newInfo = SerializationUtils.clone(info)

    newInfo.model should not be (null)
    info.model.toString() should be (newInfo.model.toString())
    info.model.parameters()._1 should be (newInfo.model.parameters()._1)
    info.model.parameters()._2 should be (newInfo.model.parameters()._2)
  }

}
