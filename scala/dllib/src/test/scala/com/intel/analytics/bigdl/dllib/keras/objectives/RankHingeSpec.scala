/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.objectives

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator
import org.scalatest.{FlatSpec, Matchers}

class RankHingeSpec extends FlatSpec with Matchers {

  "RankHinge" should "generate the correct output" in {
    val data = Array.fill(6)(RandomGenerator.RNG.uniform(0, 1).toFloat)
    val pos = data.zipWithIndex.filter(_._2 % 2 == 0).map(_._1)
    val neg = data.zipWithIndex.filter(_._2 % 2 == 1).map(_._1)
    val output = Tensor[Float](data, Array(3, 2, 1))
    val target = Tensor[Float](Array(1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f), Array(3, 2, 1))
    val actualResult = RankHinge[Float]().forward(output, target)
    val expectedArray = neg.zip(pos).map(x => x._1 - x._2 + 1.0f)
      .map(x => if (x > 0.0f) x else 0.0f)
    val expectedResult = expectedArray.sum / expectedArray.length
    require((actualResult - expectedResult).abs < 1e-6)
  }

}
