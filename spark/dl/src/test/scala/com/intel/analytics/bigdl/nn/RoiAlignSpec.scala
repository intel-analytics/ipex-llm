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

import com.intel.analytics.bigdl.nn.mkldnn.{Equivalent, Tools}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

class RoiAlignSpec extends FlatSpec with Matchers {
  "updateOutput Float" should "work properly " in {
    val spatio_scale: Float = 1.0f
    val sampling_ratio: Int = 3
    val pooled_height: Int = 2
    val pooled_width: Int = 2

    val data = Tensor[Float](
        T(T(T(0.52322268, 0.55455905, 0.92727786, 0.67067528, 0.97508526, 0.92959976),
            T(0.97565657, 0.15375137, 0.47887963, 0.17263240, 0.18203390, 0.08576220),
            T(0.56368387, 0.45025092, 0.68157887, 0.69965273, 0.17372453, 0.64042723),
            T(0.17673069, 0.79246205, 0.38627762, 0.06083101, 0.25569701, 0.14800107),
            T(0.79177606, 0.20514786, 0.28996950, 0.57044429, 0.26820791, 0.95195711),
            T(0.51260573, 0.59347671, 0.13807631, 0.73478317, 0.39982724, 0.32697338)),

          T(T(0.27678239, 0.36385250, 0.13511020, 0.91830379, 0.08621168, 0.27392197),
            T(0.26461625, 0.44157749, 0.81136179, 0.40146893, 0.27762938, 0.59823251),
            T(0.68722409, 0.74853277, 0.57244170, 0.06538194, 0.61663389, 0.14478523),
            T(0.29935873, 0.64203644, 0.27726614, 0.07251525, 0.31471008, 0.40482008),
            T(0.63593674, 0.39394510, 0.18677497, 0.17937845, 0.95044029, 0.22673655),
            T(0.66141194, 0.79792917, 0.55892295, 0.43157780, 0.91032934, 0.21400863))))
      .resize(1, 2, 6, 6)

    val rois = Array(1, 0, 0, 4, 4)
    val input = new Table
    input.insert(data)
    input.insert(Tensor(Storage(rois.map(x => x.toFloat))).resize(1, 5))

    val expected_output = Tensor[Float](
      T(T(T(T(0.38434556, 0.46265140),
            T(0.44431868, 0.50604844)),

          T(T(0.47988230, 0.40584338),
            T(0.63628036, 0.51279175)))))

    val roiAlign = RoiAlign[Float](spatio_scale, sampling_ratio, pooled_height, pooled_width)
    val output = roiAlign.forward(input)

    Equivalent.nearequals(output.asInstanceOf[Tensor[Float]], expected_output) should be(true)
  }

  "updateOutput Double" should "work properly " in {
    val spatio_scale: Float = 1.0f
    val sampling_ratio: Int = 3
    val pooled_height: Int = 2
    val pooled_width: Int = 3

    val data = Tensor[Double](Array(2, 2, 20, 30)).randn(0, 10)
    val rois = Array(1, 0, 0, 15, 10) // (img_batch, x1, y1, x2, y2)
    val input = new Table
    input.insert(data)
    input.insert(Tensor(Storage(rois.map(x => x.toDouble))).resize(1, 5))

    val roiAlign = RoiAlign[Double](spatio_scale, sampling_ratio, pooled_height, pooled_width)
    val res = roiAlign.forward(input)
  }
}
