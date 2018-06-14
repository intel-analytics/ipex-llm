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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class TFNetSpec extends FlatSpec with Matchers with BeforeAndAfter {

  var sc : SparkContext = _

  before {
    val conf = new SparkConf().setAppName("Test ObjectDetector").setMaster("local[1]")
    sc = NNContext.initNNContext(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "TFNet " should "work with different data types" in  {

    val resource = getClass().getClassLoader().getResource("tf")
    val path = resource.getPath + "/" + "multi_type_inputs_outputs.pb"

    val inputs = Seq("float_input:0", "double_input:0",
      "int_input:0", "long_input:0", "uint8_input:0")
    val outputs = Seq("float_output:0", "double_output:0",
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

    result should be (data)
    var i = 0
    while (i < 5) {
      gradInput.toTable[Tensor[Float]](i + 1).sum() should be (0.0f)
      i = i + 1
    }

  }

  "TFNet " should "be able to load from a folder" in {
    val resource = getClass().getClassLoader().getResource("tfnet")
    val net = TFNet(resource.getPath)
    val result = net.forward(Tensor[Float](4, 28, 28, 1).rand())

    result.toTensor[Float].size() should be (Array(4, 10))
  }


  "TFNet" should "should be serializable" in  {

    val resource = getClass().getClassLoader().getResource("tfnet")
    val net = TFNet(resource.getPath)
    val input = Tensor[Float](4, 28, 28, 1).rand()
    val result = net.forward(input).toTensor[Float].clone()
    val net2 = net.cloneModule()
    val result2 = net2.forward(input).toTensor[Float].clone()
    result should be (result2)
  }

  "TFNet" should "should be able to work on shrunk tensor " in  {

    val resource = getClass().getClassLoader().getResource("tfnet")
    val net = TFNet(resource.getPath)
    val input = Tensor[Float](4, 28, 28, 1).rand()
    input.resize(2, 28, 28, 1)
    val result = net.forward(input).toTensor[Float].clone()
    result.size() should be (Array(2, 10))
  }
}
