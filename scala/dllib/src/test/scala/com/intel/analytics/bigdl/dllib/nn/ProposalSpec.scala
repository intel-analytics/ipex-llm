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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest


class ProposalSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val proposal = Proposal(200, 100, Array[Float](0.1f, 0.2f, 0.3f), Array[Float](4, 5, 6))
    val score = Tensor[Float](1, 18, 20, 30).randn()
    val boxes = Tensor[Float](1, 36, 20, 30).randn()
    val imInfo = Tensor[Float](T(300, 300, 1, 1)).resize(1, 4)
    val input = T(score, boxes, imInfo)
    runSerializationTest(proposal, input)
  }
}
