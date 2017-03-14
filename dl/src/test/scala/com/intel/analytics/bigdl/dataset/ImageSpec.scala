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

package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.dataset.image.LabeledBGRImage
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class ImageSpec extends FlatSpec with Matchers {
  "image with odd width" should "flip good" in {
    val image = new LabeledBGRImage(
      Array[Float](
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        11, 12, 13, 14, 15, 16, 17, 18, 19,
        21, 22, 23, 24, 25, 26, 27, 28, 29
      ),
      3, 3, 1)

    image.hflip()
    val flippedData = Array[Float](
      7, 8, 9, 4, 5, 6, 1, 2, 3,
      17, 18, 19, 14, 15, 16, 11, 12, 13,
      27, 28, 29, 24, 25, 26, 21, 22, 23
    )

    image.content should be(flippedData)
  }

  "image with even width" should "flip good" in {
    val image = new LabeledBGRImage(
      Array[Float](
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
      ),
      4, 3, 1)

    image.hflip()
    val flippedData = Array[Float](
      10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3,
      20, 21, 22, 17, 18, 19, 14, 15, 16, 11, 12, 13,
      30, 31, 32, 27, 28, 29, 24, 25, 26, 21, 22, 23
    )

    image.content should be(flippedData)
  }
}
