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

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class SpatialFullConvolutionSpec extends FlatSpec with Matchers {

  "A SpatialFullConvolution BilinearFiller" should "generate correct parameter" in {
    val conv = new SpatialFullConvolution[Tensor[Double], Double](3, 6, 3, 3, 2, 2,
      0, 0, 0, 0, 1, false, BilinearFiller)

    val caffeWeight = Tensor(Storage(Array(
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625
    )), 1, Array(1, 3, 6, 3, 3))

    conv.weight should be (caffeWeight)
  }

  "A SpatialFullConvolution BilinearFiller(1, 2, 4, 4)" should "generate correct parameter" in {
    val conv = new SpatialFullConvolution[Tensor[Double], Double](1, 2, 4, 4, 2, 2,
      0, 0, 0, 0, 1, false, BilinearFiller)

    val caffeWeight = Tensor(Storage(Array(
      0.0625, 0.1875, 0.1875, 0.0625,
      0.1875, 0.5625, 0.5625, 0.1875,
      0.1875, 0.5625, 0.5625, 0.1875,
      0.0625, 0.1875, 0.1875, 0.0625,

      0.0625, 0.1875, 0.1875, 0.0625,
      0.1875, 0.5625, 0.5625, 0.1875,
      0.1875, 0.5625, 0.5625, 0.1875,
      0.0625, 0.1875, 0.1875, 0.0625
    )), 1, Array(1, 1, 2, 4, 4))

    conv.weight should be (caffeWeight)
  }
}
