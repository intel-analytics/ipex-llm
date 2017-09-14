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

import com.intel.analytics.bigdl.nn.{Graph, Module}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine

object Test {
  def main(args: Array[String]): Unit = {
    System.setProperty("bigdl.localMode", "true")
    Engine.init
    val modelFile = args(0)
    val model = Module.loadTF[Float](modelFile, Seq("Placeholder"), Seq("Relu_18"))
    model.asInstanceOf[Graph[Float]].saveGraphTopology("/tmp/unet")
    val imageConst = Module.loadTF[Float](modelFile, Seq(), Seq("image"))
    val image = imageConst.forward(null)
    val outputConst = Module.loadTF[Float](modelFile, Seq(), Seq("output"))
    val expectOutput = outputConst.forward(null).asInstanceOf[Tensor[Float]]

    var output = model.forward(image).asInstanceOf[Tensor[Float]]
    output.map(expectOutput, (a, b) => {
      require(math.abs(a - b) < 1e-2, s"output not match $a $b")
      a
    })
    println("Test pass")
  }
}
