/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.pipeline.api.net


import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{LayerException, T}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.SparkConf
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class TFNetSpec extends FlatSpec with Matchers with BeforeAndAfter {

  "TFNet " should "work with different data types" in {

    val resource = getClass().getClassLoader().getResource("tf")
    val path = resource.getPath + "/" + "multi_type_inputs_outputs.pb"

    val inputs = Array("float_input:0", "double_input:0",
      "int_input:0", "long_input:0", "uint8_input:0")
    val outputs = Array("float_output:0", "double_output:0",
      "int_output:0", "long_output:0", "uint8_output:0")
    val net = TFNet(path, inputs, outputs)
    val data = T(Tensor[Float](Array[Float](1.0f), Array(1, 1)),
      Tensor[Float](Array[Float](2.0f), Array(1, 1)),
      Tensor[Float](Array[Float](3.0f), Array(1, 1)),
      Tensor[Float](Array[Float](4.0f), Array(1, 1)),
      Tensor[Float](Array[Float](255.0f), Array(1, 1))
    )
    val result = net.forward(data)
    val gradInput = net.backward(data, null)

    result should be(data)
    var i = 0
    while (i < 5) {
      gradInput.toTable[Tensor[Float]](i + 1).sum() should be(0.0f)
      i = i + 1
    }

  }

  "TFNet " should "be able to load from a folder" in {
    val resource = getClass().getClassLoader().getResource("tfnet")
    val net = TFNet(resource.getPath)
    val result = net.forward(Tensor[Float](2, 4).rand())

    result.toTensor[Float].size() should be(Array(2, 2))
  }

  "TFNet" should "should be serializable by java" in {

    val resource = getClass().getClassLoader().getResource("tfnet")
    val net = TFNet(resource.getPath)
    val input = Tensor[Float](2, 4).rand()
    val result = net.forward(input).toTensor[Float].clone()
    val net2 = net.cloneModule()
    val result2 = net2.forward(input).toTensor[Float].clone()
    result should be(result2)
  }

  "TFNet" should "should be able to work on shrunk tensor " in {

    val resource = getClass().getClassLoader().getResource("tfnet")
    val net = TFNet(resource.getPath)
    val input = Tensor[Float](4, 4).rand()
    input.resize(2, 4)
    val result = net.forward(input).toTensor[Float].clone()
    result.size() should be(Array(2, 2))
  }

  "TFNet " should "work for kryo serializer" in {

    val resource = getClass().getClassLoader().getResource("tfnet")
    val net = TFNet(resource.getPath)
    val input = Tensor[Float](2, 4).rand()
    val result = net.forward(input).toTensor[Float].clone()

    val serde = new  KryoSerializer(new SparkConf()).newInstance()
    val buff = serde.serialize(net)
    val net2 = serde.deserialize[TFNet](buff)

    val result2 = net2.forward(input).toTensor[Float].clone()
    result should be(result2)
  }

  "TFNet " should "check input number" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = resource.getPath + "/" + "multi_type_inputs_outputs.pb"

    val inputs = Array("float_input:0", "double_input:0",
      "int_input:0", "long_input:0", "uint8_input:0")
    val outputs = Array("float_output:0", "double_output:0",
      "int_output:0", "long_output:0", "uint8_output:0")
    val net = TFNet(path, inputs, outputs)
    val data1 = T(Tensor[Float](Array[Float](1.0f), Array(1, 1)),
      Tensor[Float](Array[Float](2.0f), Array(1, 1)),
      Tensor[Float](Array[Float](3.0f), Array(1, 1)),
      Tensor[Float](Array[Float](4.0f), Array(1, 1))
    )
    val data2 = T(Tensor[Float](Array[Float](1.0f), Array(1, 1)))

    intercept[LayerException] {
      net.forward(data1) // this is not allowed
    }

    intercept[LayerException] {
      net.forward(data2) // this is not allowed
    }
  }

  "TFNet " should "work with backward" in {
    val resource = getClass().getClassLoader().getResource("tfnet_training")
    val net = TFNet(resource.getPath)
    val input = Tensor[Float](2, 4).rand()
    val output = net.forward(input).toTensor[Float].clone()
    val gradInput = net.backward(input, output).toTensor[Float].clone()

    gradInput.size() should be (input.size())
  }

}
