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

import com.intel.analytics.bigdl.models.vgg.VggForCifar10
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator

class VggForCifar10Spec extends KerasBaseSpec {

  "VggForCifar10 sequential" should "generate the correct outputShape" in {
    val vgg = VggForCifar10.keras(classNum = 10)
    vgg.getOutputShape().toSingle().toArray should be (Array(-1, 10))
  }

  "VggForCifar10 graph" should "generate the correct outputShape" in {
    val vgg = VggForCifar10.kerasGraph(classNum = 10)
    vgg.getOutputShape().toSingle().toArray should be (Array(-1, 10))
  }

  "VggForCifar10 sequential definition" should "be the same as graph definition" in {
    RandomGenerator.RNG.setSeed(1000)
    val kseq = VggForCifar10.keras(classNum = 10, hasDropout = false)
    RandomGenerator.RNG.setSeed(1000)
    val kgraph = VggForCifar10.kerasGraph(classNum = 10, hasDropout = false)
    val input = Tensor[Float](Array(32, 3, 32, 32)).rand()
    compareModels(kseq, kgraph, input)
  }

}
