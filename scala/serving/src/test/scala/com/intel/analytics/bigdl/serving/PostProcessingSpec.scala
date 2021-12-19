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

package com.intel.analytics.bigdl.serving

import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.serving.postprocessing.PostProcessing
import org.scalatest.{FlatSpec, Matchers}

class PostProcessingSpec extends FlatSpec with Matchers {
  "tensor to ndarray string" should "work" in {
    val t1 = Tensor(Array(4.0f, 5, 3, 4), shape = Array(2, 2))
    val ansStr = new PostProcessing(t1).tensorToNdArrayString()
    val truthStr = "[[4.0,5.0],[3.0,4.0]]"
    assert(ansStr == truthStr)
  }

  "TopN filter" should "work properly" in {
    val t1 = Tensor(Storage(Array(3.0f, 2, 1, 4, 5)))
    val ansString = new PostProcessing(t1).rankTopN(2)
    val truthString = "[[4,5.0][3,4.0]]"
    assert(ansString == truthString)
  }
  "PickTopN filter" should "work properly" in {
    val t1 = Tensor(data = Array(2.0f, 2, 3, 3), shape = Array(2, 2))
    val ansString = new PostProcessing(t1).pickTopN(1)
    val truthString = "[[2.0,2.0]]"
    assert(ansString == truthString)
  }
  "Arrow encode" should "work" in {
    val t1 = Tensor(data = Array(2.0f, 2, 3, 3), shape = Array(2, 2))
    val t2 = Tensor(data = Array(2.0f, 2, 3, 3), shape = Array(2, 2))
    (0 until 100).foreach(_ => PostProcessing(t1, "", 1))

    val byteArr = PostProcessing(t2, "", 1)
    byteArr
  }
}
