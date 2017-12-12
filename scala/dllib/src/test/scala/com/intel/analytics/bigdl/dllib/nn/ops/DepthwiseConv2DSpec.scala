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

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.utils.BigDLSpecHelper

class DepthwiseConv2DSpec extends BigDLSpecHelper {
  "DepthwiseConv2D" should "be able to save and load" in {
    val module = DepthwiseConv2D[Float](1, 1, 0, 0)
    val file = createTmpFile()
    module.saveModule(file.getAbsolutePath, overWrite = true)
    Module.loadModule[Float](file.getAbsolutePath)
  }

  "DepthwiseConv2DBackpropInput" should "be able to save and load" in {
    val module = DepthwiseConv2DBackpropInput[Float](1, 1, 0, 0, DataFormat.NHWC)
    val file = createTmpFile()
    module.saveModule(file.getAbsolutePath, overWrite = true)
    Module.loadModule[Float](file.getAbsolutePath)
  }

  "DepthwiseConv2DBackpropFilter" should "be able to save and load" in {
    val module = DepthwiseConv2DBackpropFilter[Float](1, 1, 0, 0, DataFormat.NHWC)
    val file = createTmpFile()
    module.saveModule(file.getAbsolutePath, overWrite = true)
    Module.loadModule[Float](file.getAbsolutePath)
  }
}
