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

package com.intel.analytics.bigdl.nn.mkldnn

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.nn.{Squeeze, StaticGraph, Input => NNInput}

class SqueezeSpec extends FlatSpec with Matchers {
  "a graph with squeeze" should "convert correctly" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val input = NNInput[Float]()
    val squeeze = Squeeze[Float]().inputs(input)

    val graph = new StaticGraph[Float](Array(input), Array(squeeze))

    // if there's no exception here, means it's right.
    graph.toIRgraph()
    System.clearProperty("bigdl.engineType")
  }
}
