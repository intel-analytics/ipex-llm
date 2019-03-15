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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.{Module, GaussianSampler => BGaussianSampler}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{GaussianSampler => ZGaussianSampler}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{RandomGenerator, Shape, T, Table}
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

import scala.util.Random

class GaussianSamplerSpec extends ZooSpecHelper {

  "GaussianSampler Zoo" should "be the same as BigDL" in {
    val blayer = BGaussianSampler[Float]()
    val zlayer = ZGaussianSampler[Float](inputShape = Shape(List(Shape(3), Shape(3))))
    zlayer.build(Shape(List(Shape(-1, 3), Shape(-1, 3))))
    assert(zlayer.getOutputShape() == Shape(-1, 3))
    val input = T(Tensor[Float](Array(2, 3)).rand(), Tensor[Float](Array(2, 3)).rand())
    compareOutputAndGradInputTable2Tensor(
      blayer.asInstanceOf[AbstractModule[Table, Tensor[Float], Float]],
      zlayer.asInstanceOf[AbstractModule[Table, Tensor[Float], Float]],
      input)
  }
}

class GaussianSamplerSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = ZGaussianSampler[Float](inputShape = Shape(List(Shape(3), Shape(3))))
    layer.build(Shape(List(Shape(-1, 3), Shape(-1, 3))))
    val input = T(Tensor[Float](Array(2, 3)).apply1(_ => Random.nextFloat()),
      Tensor[Float](Array(2, 3)).apply1(_ => Random.nextFloat()))

    val seed = System.currentTimeMillis()
    RandomGenerator.RNG.setSeed(seed)
    val originalOutput = layer.forward(input).asInstanceOf[Tensor[Float]].clone()
    val tmpFile = ZooSpecHelper.createTmpFile()
    val absPath = tmpFile.getAbsolutePath
    layer.saveModule(absPath, overWrite = true)
    val loadedLayer = Module.loadModule[Float](absPath)
    RandomGenerator.RNG.setSeed(seed)
    val loadedOutput = loadedLayer.forward(input).asInstanceOf[Tensor[Float]].clone()
    originalOutput.asInstanceOf[Tensor[Float]].size.sameElements(
      loadedOutput.asInstanceOf[Tensor[Float]].size) should be (true)
    originalOutput.asInstanceOf[Tensor[Float]].
      almostEqual(loadedOutput.asInstanceOf[Tensor[Float]], 1e-6) should be (true)
  }
}
