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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.dllib.utils.RandomGenerator._
import com.intel.analytics.bigdl.dllib.utils.TestUtils
import com.intel.analytics.bigdl.dllib.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class SpatialWithinChannelLRNSpec extends FlatSpec with Matchers{
  "forward" should "work" in {
    val layer = new SpatialWithinChannelLRN[Float](5, 5e-4, 0.75)
    val input = Tensor(Storage(Array(-0.5629349351, 0.1707911491, -0.6980619431, 0.7621926665,
      0.5278353691, -0.3644135892, -0.7721814513, 0.1747673303,
      -0.0607845485, -0.6372885108, -0.8762480021, -0.1419942826,
      0.1868573427, 0.9514402151, -1.3372359276, -2.2525136471,
      0.9049832225, -0.5155935884, 1.7007764578, 0.1731968075,
      -0.8370018601, 1.0927723646, -1.9021866322, 1.7037003040,
      -0.4785003364, 2.9398944378, 1.0938369036, -1.1450070143,
      -0.4233447611, -0.0770315304, 0.3048466742, -0.3763152659,
      -0.4508697689, -1.5802454948, -0.2301389128, -0.0077587459,
      1.3115756512, -0.8771016598, 0.2766959071, 0.9347394705,
      -0.3583759665, -0.0234385580, -0.8723886013, 1.0725305080,
      -0.5435923934, -1.2200671434, 1.1134344339, -0.3956952393,
      1.2290611267, -0.0300420318, -0.6727856994, -1.2889897823,
      0.8753286004, 1.6655322313, 0.0144331316, 0.1103931740,
      -1.3110619783, -0.0822269097, -1.0805166960, -1.7619274855,
      0.6417796016, -0.1190131158, 0.2828691006, 0.2091746032,
      0.2575667202, 0.0919162408, -0.4426752627, -0.8307037950,
      -0.4315507710, 0.6092016101, 0.1071894467, 0.3243490458,
      0.2481118143, -0.9265315533, -1.1380627155, 0.2642920315,
      0.4124689102, 0.0631492808, 0.2114779800, -0.8970818520,
      1.3339444399, -1.9507933855, -2.0102806091, -0.4132987857,
      1.7795655727, 0.7749545574, 0.1243268624, 0.2292353660,
      -0.2117905766, -0.7013441324, -0.1906698346, 0.7260479927,
      -0.1027618498, 0.4159305096, -0.9893618822, 0.9117889404,
      -0.4076909125, -0.5034028888, 0.2496513128, 1.1489638090,
      0.0799294263, -1.1163944006, 1.0516333580, 0.4432829320,
      1.4964165688, 0.4551719129, 1.2555737495, -0.7395679355,
      0.8859211802, 1.4474883080, -1.8880417347, 0.2351293117,
      -1.4667420387, -1.3699244261, -1.9512387514, -0.2791400254,
      0.1369622797, -0.8618115783, 0.5658946037, -1.0372757912,
      0.8631318808, -0.2621137798, 1.1617847681, 0.8746930361,
      -0.3064057231, -1.6209356785, -1.2230005264, 0.3100630939,
      0.8659827113, 0.6965786815, 1.9879816771, -0.4788473845,
      0.9765572548, -0.1230701804, 0.0977652520, -0.5547776818,
      0.2019653022, 1.4806493521, -0.6643462181, -0.2502589226,
      -0.3229486048, -0.3250016570, -0.8827708960, -0.3872688115,
      -1.3991631269, -0.1658344567, 0.9363160133, -1.0897105932,
      0.0178568624, 0.6493149996, -0.7701879740, -0.5503393412,
      0.2017845362, -1.2033561468, 0.3187648654, -1.8862437010,
      -1.0454750061, 0.0067651863, -0.6173903942, -1.6129561663,
      -0.0125843389, -0.0205810498, 0.0797127113, 0.5228791833,
      0.0640839785, -0.2918730080, 0.4691511989, -0.4645946920).map(_.toFloat)))
      .resize(1, 4, 7, 6)

    val expected = Tensor(Storage(Array(-0.5628995895, 0.1707648933, -0.6979351044, 0.7620602250,
      0.5277513266, -0.3643679619, -0.7720909715, 0.1747278422,
      -0.0607658178, -0.6370970607, -0.8759979010, -0.1419600844,
      0.1868072301, 0.9510628581, -1.3365921974, -2.2514543533,
      0.9046882987, -0.5154578686, 1.7003303766, 0.1731241792,
      -0.8365827203, 1.0922356844, -1.9015303850, 1.7032119036,
      -0.4783609509, 2.9385557175, 1.0932601690, -1.1444340944,
      -0.4231994748, -0.0770096704, 0.3047703505, -0.3761879206,
      -0.4506902993, -1.5796643496, -0.2300873548, -0.0077572609,
      1.3113185167, -0.8768681288, 0.2766207457, 0.9345140457,
      -0.3583408892, -0.0234367829, -0.8723114133, 1.0723847151,
      -0.5434926748, -1.2197744846, 1.1131868362, -0.3956218362,
      1.2289429903, -0.0300377030, -0.6726560593, -1.2886765003,
      0.8751313090, 1.6652204990, 0.0144315129, 0.1103748828,
      -1.3107807636, -0.0822052434, -1.0802621841, -1.7615847588,
      0.6417075396, -0.1189959049, 0.2828189433, 0.2091255486,
      0.2575123906, 0.0919020027, -0.4426211119, -0.8305486441,
      -0.4314348996, 0.6090135574, 0.1071601808, 0.3242850900,
      0.2480878979, -0.9263827205, -1.1378066540, 0.2642342746,
      0.4123934209, 0.0631408766, 0.2114592046, -0.8969451189,
      1.3336566687, -1.9503731728, -2.0099184513, -0.4132445455,
      1.7794363499, 0.7748803496, 0.1243130341, 0.2292126119,
      -0.2117739469, -0.7012900114, -0.1906458735, 0.7259376645,
      -0.1027422175, 0.4158595204, -0.9892163873, 0.9116867185,
      -0.4076002836, -0.5032773018, 0.2495712340, 1.1485998631,
      0.0799085051, -1.1162009239, 1.0513975620, 0.4431669712,
      1.4959123135, 0.4550207257, 1.2552160025, -0.7394225597,
      0.8857015371, 1.4470622540, -1.8873686790, 0.2350427955,
      -1.4662616253, -1.3696243763, -1.9507690668, -0.2790654302,
      0.1369171739, -0.8615318537, 0.5657315850, -1.0370885134,
      0.8629699349, -0.2620584667, 1.1614948511, 0.8744715452,
      -0.3063384593, -1.6206996441, -1.2229285240, 0.3100406528,
      0.8658581376, 0.6964817047, 1.9877101183, -0.4787881970,
      0.9764578342, -0.1230537966, 0.0977452397, -0.5546795130,
      0.2019301802, 1.4804306030, -0.6642692685, -0.2502166629,
      -0.3228704631, -0.3249176145, -0.8825492859, -0.3871819377,
      -1.3990192413, -0.1658033431, 0.9361273646, -1.0894701481,
      0.0178530309, 0.6491894722, -0.7701167464, -0.5502436757,
      0.2017461210, -1.2031224966, 0.3187061250, -1.8859360218,
      -1.0453878641, 0.0067640827, -0.6172866821, -1.6126719713,
      -0.0125822360, -0.0205780119, 0.0797094926, 0.5228255987,
      0.0640771016, -0.2918325961, 0.4690902829, -0.4645373523).map(_.toFloat)))
      .resize(1, 4, 7, 6)

    val out = layer.forward(input)

    out.map(expected, (a, b) => {
      TestUtils.conditionFailTest(Math.abs(a - b) < 1e-6);
      a
    })
  }

  "gradient check" should "pass" in {
    val layer = SpatialWithinChannelLRN[Double](5, 5e-4, 0.75)
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](4, 4, 4, 6).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer[Double](layer, input, 1e-3) should be(true)
  }
}

class SpatialWithinChannelLRNSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val spatialWithinChannelLRN = new SpatialWithinChannelLRN[Float](5, 5e-4, 0.75).
      setName("spatialWithinChannelLRN")
    val input = Tensor[Float](1, 4, 7, 6).apply1( e => Random.nextFloat())
    runSerializationTest(spatialWithinChannelLRN, input)
  }
}
