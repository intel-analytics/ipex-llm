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

package com.intel.analytics.bigdl.keras

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor

class HighwaySpec extends KerasBaseSpec {
  "highway forward backward" should "work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[2])
        |input = np.random.uniform(0, 1, [3, 2])
        |output_tensor = Highway(activation='tanh')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val highway = Highway[Float](2, activation = Tanh[Float])
    def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] =
      Array(in(1).t(), in(3), in(0).t(), in(2))
    checkHighwayOutputAndGrad(highway, kerasCode, weightConverter)
  }

  "highway forward backward noBias" should "work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[2])
        |input = np.random.uniform(0, 1, [3, 2])
        |output_tensor = Highway(activation='tanh', bias=None)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val highway = Highway[Float](2, activation = Tanh[Float], withBias = false)
    def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] =
      Array(in(1).t(), in(0).t())

    checkHighwayOutputAndGrad(highway, kerasCode, weightConverter)
  }

  "highway forward backward no activation" should "work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[2])
        |input = np.random.uniform(0, 1, [3, 2])
        |output_tensor = Highway(bias=None)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val highway = Highway[Float](2, withBias = false)
    def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] =
      Array(in(1).t(), in(0).t())

    checkHighwayOutputAndGrad(highway, kerasCode, weightConverter)
  }

  def checkHighwayOutputAndGrad(bmodel: Graph[Float],
    kerasCode: String,
    weightConverter: (Array[Tensor[Float]]) => Array[Tensor[Float]],
    precision: Double = 1e-5): Unit = {
    ifskipTest()
    val (gradInput, gradWeight, weights, input, target, output) = KerasRunner.run(kerasCode)
    // Ensure they share the same weights
    if (weights != null) {
      bmodel.setWeightsBias(weightConverter(weights))
    }
    val boutput = bmodel.forward(input).toTensor[Float]
    boutput.almostEqual(output, precision) should be(true)

    val bgradInput = bmodel.backward(input, boutput.clone()).toTensor[Float]
    bgradInput.almostEqual(gradInput, precision) should be(true)
  }

  "Highway serializer" should "work properly" in {
    val module = Highway[Float](2, activation = Tanh[Float])

    val input = Tensor[Float](3, 2).randn()
    val res1 = module.forward(input.clone()).toTensor[Float].clone()
    val clone = module.cloneModule()
    val tmpFile = java.io.File.createTempFile("module", ".bigdl")
    module.saveModule(tmpFile.getAbsolutePath, null, true)
    val loaded = Module.loadModule[Float](tmpFile.getAbsolutePath)

    val res2 = loaded.forward(input.clone())
    val namedModule = Utils.getNamedModules[Float](clone)
    val namedModule2 = Utils.getNamedModules[Float](loaded)
    res1 should be(res2)
    if (tmpFile.exists()) {
      tmpFile.delete()
    }
  }
}
