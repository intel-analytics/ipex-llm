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
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

class RoiAlignSpec extends FlatSpec with Matchers {
  "updateOutput Float" should "work properly " in {
    val spatio_scale: Float = 1.0f
    val sampling_ratio: Int = 3
    val pooled_height: Int = 2
    val pooled_width: Int = 3

    val data = Tensor[Float](Array(2, 2, 20, 30)).randn(0, 10)
    val rois = Array(1, 0, 0, 15, 10) // , 1, 6, 2, 12, 8, 1, 3, 1, 13, 7, 0, 3, 3, 3, 3)
    val input = new Table
    input.insert(data)
    input.insert(Tensor(Storage(rois.map(x => x.toFloat))).resize(1, 5))

    val roiAlign = RoiAlign[Float](spatio_scale, sampling_ratio, pooled_height, pooled_width)
    val res = roiAlign.forward(input)
  }

  "updateOutput Double" should "work properly " in {
    val spatio_scale: Float = 1.0f
    val sampling_ratio: Int = 3
    val pooled_height: Int = 2
    val pooled_width: Int = 3

    val data = Tensor[Double](Array(2, 2, 20, 30)).randn(0, 10)
    val rois = Array(1, 0, 0, 15, 10) // , 1, 6, 2, 12, 8, 1, 3, 1, 13, 7, 0, 3, 3, 3, 3)
    val input = new Table
    input.insert(data)
    input.insert(Tensor(Storage(rois.map(x => x.toDouble))).resize(1, 5))

    val roiAlign = RoiAlign[Double](spatio_scale, sampling_ratio, pooled_height, pooled_width)
    val res = roiAlign.forward(input)
  }
}
