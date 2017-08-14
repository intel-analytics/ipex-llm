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

package com.intel.analytics.bigdl.torch
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers, ParallelTestExecution}

import scala.sys.process._


class TorchSpec extends FlatSpec with BeforeAndAfter with Matchers with ParallelTestExecution {
  def torchCheck(): Unit = {
    var hasTorch = true

    val torchPath = System.getProperty("torch_location")
    // Skip on windows
    if (System.getProperty("os.name").toLowerCase().contains("win")) {
      hasTorch = false
    }
    val exitValue = if (torchPath != null) s"ls $torchPath".! else "which th".!

    if (exitValue != 0) {
      hasTorch = false
    }
    if (!hasTorch) {
      cancel("Torch is not installed")
    }
  }
}
