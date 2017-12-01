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

package com.intel.analytics.bigdl.transform.vision.image.util

import com.intel.analytics.bigdl.tensor.Tensor

object BboxUtil {
  def decodeRois(output: Tensor[Float]): Tensor[Float] = {
    // ignore if decoded
    if (output.nElement() < 6 || output.dim() == 2) return output
    val num = output.valueAt(1).toInt
    require(num >= 0)
    if (num == 0) {
      Tensor[Float]()
    } else {
      output.narrow(1, 2, num * 6).view(num, 6)
    }
  }

  // inplace scale
  def scaleBBox(bboxes: Tensor[Float], height: Float, width: Float): Unit = {
    if (bboxes.nElement() == 0) return
    bboxes.select(2, 1).mul(width)
    bboxes.select(2, 2).mul(height)
    bboxes.select(2, 3).mul(width)
    bboxes.select(2, 4).mul(height)
  }
}
