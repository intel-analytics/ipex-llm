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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class NormalizeSpec extends FlatSpec with Matchers {
  "Normalize with 4d" should "work properly" in {
    val input = Tensor(Storage(Array(
      0.5507978797, 0.7081478238, 0.2909047306, 0.5108276010,
      0.8929469585, 0.8962931037, 0.1255853176, 0.2072428763,
      0.0514672026, 0.4408098459, 0.0298762117, 0.4568332136,
      0.6491440535, 0.2784872949, 0.6762549281, 0.5908628106,
      0.0239818823, 0.5588541031, 0.2592524588, 0.4151012003,
      0.2835250795, 0.6931379437, 0.4404537082, 0.1568677425,
      0.5446490049, 0.7803147435, 0.3063635230, 0.2219578773,
      0.3879712522, 0.9363836646, 0.9759954214, 0.6723836660,
      0.9028341174, 0.8457508683, 0.3779940307, 0.0922170058)
      .map(x => x.toFloat))).resize(2, 3, 2, 3)

    val expectedOutput = Tensor(Storage(Array(
      12.8011369705, 17.9583473206, 7.8839497566, 11.3913326263,
      19.9816169739, 15.5767812729, 2.9187383652, 5.2555966377,
      1.3948376179, 9.8299531937, 0.6685447693, 7.9393572807,
      15.0868082047, 7.0623264313, 18.3275127411, 13.1760978699,
      0.5366464257, 9.7123899460, 4.5191359520, 7.4756622314,
      5.7009272575, 12.4241008759, 12.6179037094, 3.2889757156,
      9.4940004349, 14.0528850555, 6.1601471901, 3.9784679413,
      11.1144113541, 19.6327362061, 17.0129756927, 12.1091270447,
      18.1535682678, 15.1595993042, 10.8285894394, 1.9334726334)
      .map(x => x.toFloat))).resize(2, 3, 2, 3)


    val normalizer = Normalize[Float](2)
    val output = normalizer.forward(input)
    val mul = CMul[Float](size = Array(1, 3, 1, 1))
    mul.weight.setValue(1, 1, 1, 1, 20)
    mul.weight.setValue(1, 2, 1, 1, 20)
    mul.weight.setValue(1, 3, 1, 1, 20)
    mul.forward(output) should be(expectedOutput)
  }

  "normalize with more data" should "work properly" in {
    val input = Tensor(Storage(Array(
      0.5507978797, 0.7081478238, 0.2909047306, 0.5108276010,
      0.8929469585, 0.8962931037, 0.1255853176, 0.2072428763,
      0.0514672026, 0.4408098459, 0.0298762117, 0.4568332136,
      0.6491440535, 0.2784872949, 0.6762549281, 0.5908628106,
      0.0239818823, 0.5588541031, 0.2592524588, 0.4151012003,
      0.2835250795, 0.6931379437, 0.4404537082, 0.1568677425,
      0.5446490049, 0.7803147435, 0.3063635230, 0.2219578773,
      0.3879712522, 0.9363836646, 0.9759954214, 0.6723836660,
      0.9028341174, 0.8457508683, 0.3779940307, 0.0922170058,
      0.6534109116, 0.5578407645, 0.3615647554, 0.2250545025,
      0.4065199196, 0.4689402580, 0.2692355812, 0.2917927802,
      0.4576863945, 0.8605338931, 0.5862529278, 0.2834878564,
      0.2779774964, 0.4546220899, 0.2054103464, 0.2013787180,
      0.5140350461, 0.0872293711, 0.4835855365, 0.3621762097,
      0.7076866031, 0.7467462420, 0.6910929084, 0.6891804338,
      0.3736001253, 0.6681348085, 0.3398486674, 0.5727938414,
      0.3258071542, 0.4451450408, 0.0615289323, 0.2426754236,
      0.9716026187, 0.2305842042, 0.6914775372, 0.6504768729,
      0.7239391208, 0.4750885963, 0.5966637731, 0.0669694245,
      0.0725621358, 0.1989760250, 0.1518609971, 0.1001043469,
      0.1292938590, 0.5532777309, 0.1878148317, 0.9521012306,
      0.6816117764, 0.5410196781, 0.7071806192, 0.2638866603,
      0.9267256856, 0.8391930461)
      .map(x => x.toFloat))).reshape(Array(2, 5, 3, 3))

    val expectedOutput = Tensor(Storage(Array(
      10.7962512970, 13.2859182358, 4.9504461288, 6.9418644905,
      13.7528429031, 11.7681846619, 2.0863511562, 4.4173331261,
      1.3001513481, 8.6403636932, 0.5605226755, 7.7741193771,
      8.8215084076, 4.2891592979, 8.8791189194, 9.8160142899,
      0.5111681819, 14.1176309586, 5.0816369057, 7.7879228592,
      4.8248634338, 9.4193611145, 6.7837071419, 2.0596482754,
      9.0482635498, 16.6322250366, 7.7392773628, 4.3506212234,
      7.2789239883, 15.9348278046, 13.2632369995, 10.3558073044,
      11.8540668488, 14.0504741669, 8.0568532944, 2.3295626640,
      12.8075809479, 10.4659318924, 6.1528968811, 3.0583662987,
      6.2610712051, 6.1571102142, 4.4728155136, 6.2194943428,
      11.5619573593, 11.7851772308, 12.7779006958, 3.9678306580,
      5.2822084427, 8.9109497070, 2.9043090343, 7.0540165901,
      7.1670479774, 1.5497876406, 6.6227970123, 7.8939504623,
      9.9051179886, 14.1898860931, 13.5459632874, 9.7443618774,
      13.0866928101, 9.3156175613, 6.0380268097, 7.8445219994,
      7.1012549400, 6.2304615974, 1.1691904068, 4.7566289902,
      13.7375459671, 8.0770444870, 9.6410789490, 11.5568981171,
      9.9144859314, 10.3549757004, 8.3511896133, 1.2725722790,
      1.4222748280, 2.8133335114, 5.3194799423, 1.3957271576,
      2.2971394062, 7.5772452354, 4.0935902596, 13.3260612488,
      12.9521827698, 10.6044101715, 9.9988679886, 9.2435827255,
      12.9210777283, 14.9097824097)
      .map(x => (x / 20).toFloat))).resize(2, 5, 3, 3)


    val normalizer = Normalize[Float](2)
    val output = normalizer.forward(input)

    output should be(expectedOutput)
  }

  "Normalize 4d backward" should "work properly" in {
    val seed = 100
    RNG.setSeed(seed)
    val layer = Normalize[Double](2)
    val input = Tensor[Double](3, 3, 8, 8).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer(layer, input, 1e-3) should be(true)
  }
}

class NormalizeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val normalizer = Normalize[Float](2).setName("normalizer")
    val input = Tensor[Float](2, 3, 4, 4).apply1(e => Random.nextFloat())
    runSerializationTest(normalizer, input)
  }
}
