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
package com.intel.analytics.bigdl.dllib.utils.tf.loaders

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.T
import com.intel.analytics.bigdl.dllib.utils.serializer.ModuleSerializationTest

import java.security.SecureRandom

class SplitLoadTFSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val splitLoadTF = new SplitLoadTF[Float](1).setName("splitLoadTD")
    val input = T(Tensor[Int](T(1)),
      Tensor[Float](1, 6, 2).apply1(_ => new SecureRandom().nextFloat())
    )
    runSerializationTest(splitLoadTF, input)
  }
}
