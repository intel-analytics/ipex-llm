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

package com.intel.analytics.bigdl.dllib.nn

import breeze.numerics.abs
import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class SoftmaxWithCriterionSpec extends FlatSpec with Matchers {
  val inputArr = Array(-3.8623790740966796875, -5.576374053955078125,
    10.298772811889648438, 9.0803890228271484375,
    1.3665539026260375977, -0.44133603572845458984,
    -9.40171051025390625, 1.0564124584197998047, 13.553049087524414062,
    -13.990137100219726562, 0.38796663284301757812, 1.6085460186004638672,
    8.8876256942749023438, 2.3242428302764892578,
    -4.9687619209289550781, 3.7455892562866210938, 2.0669219493865966797,
    19.429233551025390625, 7.1232995986938476562,
    -10.957750320434570312, 4.5843319892883300781, 16.586359024047851562,
    -1.0300438404083251953, -21.75362396240234375,
    -2.7482614517211914062, 2.2115952968597412109, 0.85470116138458251953,
    1.8852581977844238281, -0.88053613901138305664, -21.679836273193359375)
  val targetArr = Array(2, 4, 2, 4, 1, 2)
  val input = Tensor(Storage(inputArr.map(x => x.toFloat))).resize(1, 5, 2, 3)
  val target = Tensor(Storage((targetArr).map(x => x.toFloat))).resize(1, 1, 2, 3)

  "SoftmaxWithCriterion forward" should "work properly" in {
    val normMode = NormMode.apply(2)
    val sfmLoss = new SoftmaxWithCriterion[Float](normalizeMode = normMode)
    val actOut = sfmLoss.forward(input, target)
    var sum = 0f
    for (tar <- 1 to 5) {
      val res = new SoftmaxWithCriterion[Float](ignoreLabel = Some(tar),
        normalizeMode = normMode).forward(input, target)
      sum += res
    }
    assert(abs(actOut - 49.084717) < 1e-4)
    assert(abs(actOut * 4 - sum) < 1e-4)
  }

  "SoftmaxWithCriterion backward" should "work properly" in {
    val normMode = NormMode.apply(1)
    val sfmLoss = new SoftmaxWithCriterion[Float](normalizeMode = normMode, ignoreLabel = Some(1))
    val actOut = sfmLoss.forward(input, target)
    assert(abs(actOut - 8.274073) < 1e-4)

    val actGradInput = sfmLoss.backward(input, target)

    val expectedGradInput = Array(1.4155314E-7, 2.5500047E-8, 0.19999984,
      0.19989611, 0.0, 1.46410375E-5,
      -0.2, 7.4783907E-7, -7.3909763E-7,
      2.596081E-8, 0.0, -0.045566916,
      0.19971798, 2.818228E-4, 1.9171863E-7,
      3.0882305E-8, 0.0, 0.19999996,
    0.18536577, -0.2, 0.014634231,
    0.0, 0.0, 4.4687676E-18,
    0.001109384, 0.15816866, 0.040721968,
    0.18815985, 0.0, 1.0973286E-11)

    assert(expectedGradInput.length == actGradInput.nElement())
    (actGradInput.storage().array() zip expectedGradInput).foreach(x => {
      // because in caffe, they use weight 2 for the loss
      assert(abs(x._1 - x._2) < 1e-4)
    })
  }
}
