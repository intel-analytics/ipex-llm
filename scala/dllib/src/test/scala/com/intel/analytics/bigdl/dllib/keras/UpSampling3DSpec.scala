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
import com.intel.analytics.bigdl.utils.{Shape, TestUtils}

class UpSampling3DSpec extends KerasBaseSpec {
  "UpSampling3D forward with size 1" should "work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[1, 2, 3, 4])
        |input = np.random.uniform(0, 1, [2, 1, 2, 3, 4])
        |output_tensor = UpSampling3D((1, 1, 1))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val model = UpSampling3D[Float](Array(1, 1, 1))
    checkOutputAndGrad(model, kerasCode)
  }

  "UpSampling3D forward with size 2" should "work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[1, 1, 2, 4])
        |input = np.random.uniform(0, 1, [2, 1, 1, 2, 4])
        |output_tensor = UpSampling3D((2, 2, 2), dim_ordering = 'th')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val model = UpSampling3D[Float](Array(2, 2, 2))
    checkOutputAndGrad(model, kerasCode)
  }

  "UpSampling3D forward with size 2, 3, 4" should "work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[2, 3, 2, 4])
        |input = np.random.uniform(0, 1, [2, 2, 3, 2, 4])
        |output_tensor = UpSampling3D((2, 3, 4), dim_ordering = 'th')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val model = UpSampling3D[Float](Array(2, 3, 4))
    checkOutputAndGrad(model, kerasCode)
  }

  "UpSampling3D serializer" should "work properly" in {
    val module = UpSampling3D[Float](Array(2, 2, 2))

    val input = Tensor[Float](1, 2, 2, 2, 2).randn()
    val res1 = module.forward(input).clone()
    val tmpFile = java.io.File.createTempFile("module", ".bigdl")
    module.saveModule(tmpFile.getAbsolutePath, null, true)
    val loaded = Module.loadModule[Float](tmpFile.getAbsolutePath)
    val res2 = loaded.forward(input)
    res1 should be(res2)
    if (tmpFile.exists()) {
      tmpFile.delete()
    }
  }

  "UpSampling3D computeOutputShape" should "work properly" in {
    val layer = UpSampling3D[Float](Array(2, 1, 3))
    TestUtils.compareOutputShape(layer, Shape(3, 8, 12, 8)) should be (true)
  }

}
