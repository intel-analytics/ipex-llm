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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest


class SelectTensorSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val transformer = (TensorOp[Float]() ** 3 * 4.5f).ceil
    val select = SelectTensor(Tensor.scalar("2"), transformer)
    val t1 = Tensor[Float](3, 4).randn()
    val t2 = Tensor[Float](2, 3).randn()
    val input = T().update(Tensor.scalar(1), t1).update(Tensor.scalar("2"), t2)
    runSerializationTest(select, input)
  }
}
