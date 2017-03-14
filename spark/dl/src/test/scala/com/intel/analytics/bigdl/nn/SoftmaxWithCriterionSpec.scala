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

import breeze.numerics.abs
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
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
    assert(abs(actOut - 51.643194557149605828) < 1e-4)
    assert(abs(actOut * 4 - sum) < 1e-4)
  }

  "SoftmaxWithCriterion backward" should "work properly" in {
    val normMode = NormMode.apply(1)
    val sfmLoss = new SoftmaxWithCriterion[Float](normalizeMode = normMode, ignoreLabel = Some(1))
    val actOut = sfmLoss.forward(input, target)
    assert(abs(actOut - 10.073171615600585938) < 1e-4)

    val actGradInput = sfmLoss.backward(input, target)

    val expectedGradInput = Array(9.9112855878047412261e-07,
      6.813194340793415904e-05, 0.014867544174194335938,
      0.00021979543089400976896, 0, 9.3838581349814376154e-10,
      -0.40000000596046447754, 0.051752727478742599487,
      -0.014917778782546520233, 2.1019214065673766378e-14, 0,
      -0.40000000596046447754, 0.34149685502052307129,
      0.18388444185256958008, 3.4804334969606998129e-09,
      1.0596063475531991571e-06, 0, 0.40000000596046447754,
      0.058499157428741455078, -0.39999970793724060059,
      4.9033053073799237609e-05, -0.0002209901867900043726,
      0, 5.2068160617220955731e-19, 3.0198482363630319014e-06,
      0.16429440677165985107, 1.1768768217734759673e-06,
      1.6489701692989910953e-07, 0, 5.6055445173304457315e-19)

    assert(expectedGradInput.length == actGradInput.nElement())
    (actGradInput.storage().array() zip expectedGradInput).foreach(x => {
      // because in caffe, they use weight 2 for the loss
      assert(abs(x._1 - x._2 / 2.0) < 1e-4)
    })
  }
}
