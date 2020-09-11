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
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.BeforeAndAfter
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers._

import scala.util.Random

class ScaleSpec extends AnyFlatSpec with should.Matchers{
  val input = Tensor(Storage(Array(
    1.3968338966, 1.0623255968, 0.0113903601, 1.6713322401,
    1.2211480141, -1.4509917498, 0.9407374859, 1.3669412136,
    -0.1955126524, -0.1213591248, -0.4367840588, 1.0001722574,
    0.9071449637, 0.9841009378, 1.0343499184, -1.2177716494,
    0.0444964543, -0.3394794762, 0.9685149193, 0.0013315412,
    1.2855026722, -1.0687378645, -0.8125880957, 0.2460595369,
    -0.5525790453, -1.4602638483, -0.6113839149, -0.4403405488,
    0.8861535788, -1.1445546150, 0.3850491047, 0.2242770344,
    1.5059198141, -0.3505563736, 0.3127737045, 0.7735480666,
    0.5683772564, -1.3868474960, 1.3575958014, -1.9670666456,
    0.9238662124, 1.6204817295, 0.4601316452, 0.6467041969,
    0.8394199014, -0.9312202334, 0.5268830657, -0.0692716911,
    -0.8119943738, 0.3967324793, -0.4461912215, -0.1913439631,
    1.0490620136, -0.0018771883, -1.5796754360, 0.4819125235,
    0.1086308882, 0.9048879743, 1.8832274675, 0.2517957985,
    -0.1036709696, -0.4925992191, 0.0656546056, -1.0584318638,
    0.2286393940, 0.7230707407, 0.8117146492, 0.8037125468,
    -1.5480978489, 0.6889602542, 1.9156422615, -1.3656581640,
    -0.1560046375, -0.2486510724, -0.0217402168, 1.2162036896,
    0.6996396780, 0.6564534903, 0.8134143949, 0.0601990744,
    1.0776667595, -0.5910157561, 1.0320621729, 0.0460616127,
    -1.3710715771, -0.3773273826, 0.3672361672, 0.5238098502,
    0.0119323730, -0.2517851293, 1.4480800629, 0.0828312412,
    1.6660629511, -0.6199936271, -1.7660367489, 0.3837936223,
    -0.7086514831, 1.6811115742, 0.6771110296, 1.1161595583,
    0.1626710445, 0.4269112945, -0.4675120413, 1.5107829571,
    0.6962547302, 0.0728486627, -0.3824745715, 0.5898073316,
    0.1672963798, 1.2455582619, 2.1523268223, 1.0041118860,
    -0.5187900066, -0.6860033870, -0.6559141874, 0.3032713532,
    -2.0993845463, 0.2686079144, -0.4012460411, 0.6594560742))).resize(1, 4, 5, 6)

  "scale forward backward" should "work properly" in {
    val scale = new Scale[Double](Array(1, 4, 1, 1))
    scale.parameters()._1(0).copy(Tensor(Storage(Array(0.4, 0.3, 0.2, 0.1)))) // weight
    scale.parameters()._1(1).copy(Tensor(Storage(Array(0.1, 0.01, 0.03, 0.04)))) // bias

    val expectedOutput = Tensor(Storage(Array(
      0.6587336063, 0.5249302387, 0.1045561433, 0.7685329318,
      0.5884591937, -0.4803967178, 0.4762949944, 0.6467764974,
      0.0217949376, 0.0514563508, -0.0747136250, 0.5000689030,
      0.4628579915, 0.4936403632, 0.5137400031, -0.3871086836,
      0.1177985817, -0.0357917920, 0.4874059558, 0.1005326211,
      0.6142011285, -0.3274951577, -0.2250352502, 0.1984238178,
      -0.1210316196, -0.4841055572, -0.1445535719, -0.0761362240,
      0.4544614255, -0.3578218520, 0.1255147308, 0.0772831142,
      0.4617759585, -0.0951669216, 0.1038321108, 0.2420644313,
      0.1805131882, -0.4060542881, 0.4172787368, -0.5801200271,
      0.2871598601, 0.4961445332, 0.1480395049, 0.2040112764,
      0.2618259788, -0.2693660855, 0.1680649370, -0.0107815079,
      -0.2335983217, 0.1290197521, -0.1238573715, -0.0474031940,
      0.3247185946, 0.0094368430, -0.4639026523, 0.1545737684,
      0.0425892696, 0.2814663947, 0.5749682784, 0.0855387375,
      0.0092658047, -0.0685198456, 0.0431309193, -0.1816863716,
      0.0757278800, 0.1746141464, 0.1923429370, 0.1907425076,
      -0.2796195745, 0.1677920520, 0.4131284654, -0.2431316376,
      -0.0012009293, -0.0197302159, 0.0256519560, 0.2732407451,
      0.1699279398, 0.1612907052, 0.1926828772, 0.0420398153,
      0.2455333620, -0.0882031545, 0.2364124358, 0.0392123237,
      -0.2442143261, -0.0454654768, 0.1034472361, 0.1347619742,
      0.0323864743, -0.0203570258, 0.1848080158, 0.0482831225,
      0.2066062987, -0.0219993629, -0.1366036832, 0.0783793628,
      -0.0308651477, 0.2081111670, 0.1077111065, 0.1516159475,
      0.0562671050, 0.0826911330, -0.0067512058, 0.1910783052,
      0.1096254736, 0.0472848639, 0.0017525405, 0.0989807323,
      0.0567296371, 0.1645558178, 0.2552326918, 0.1404111981,
      -0.0118790008, -0.0286003426, -0.0255914181, 0.0703271329,
      -0.1699384451, 0.0668607950, -0.0001246072, 0.1059456095))).resize(1, 4, 5, 6)

    val output = scale.forward(input)
    output.map(expectedOutput, (a, b) => {
      assert(Math.abs(a - b) < 1e-6)
      a
    })
    val outDiff = Tensor(Storage(Array(0.2203191519, 0.2832591236, 0.1163618937, 0.2043310404,
      0.3571787775, 0.3585172594, 0.0502341278, 0.0828971490,
      0.0205868818, 0.1763239354, 0.0119504845, 0.1827332824,
      0.2596576214, 0.1113949195, 0.2705019712, 0.2363451272,
      0.0095927529, 0.2235416472, 0.1037009880, 0.1660404801,
      0.1134100333, 0.2772551775, 0.1761814803, 0.0627470985,
      0.2178596109, 0.3121258914, 0.1225454137, 0.0887831524,
      0.1551885009, 0.3745534718, 0.2927986383, 0.2017151117,
      0.2708502412, 0.2537252605, 0.1133982167, 0.0276651029,
      0.1960232854, 0.1673522294, 0.1084694341, 0.0675163567,
      0.1219559833, 0.1406820863, 0.0807706788, 0.0875378400,
      0.1373059303, 0.2581601739, 0.1758758873, 0.0850463584,
      0.0833932534, 0.1363866329, 0.0616231076, 0.0604136176,
      0.1542105228, 0.0261688121, 0.1450756639, 0.1086528674,
      0.2123059928, 0.2240238786, 0.2073278874, 0.2067541331,
      0.0747200251, 0.1336269677, 0.0679697320, 0.1145587713,
      0.0651614293, 0.0890290067, 0.0123057868, 0.0485350862,
      0.1943205297, 0.0461168401, 0.1382955164, 0.1300953776,
      0.1447878331, 0.0950177237, 0.1193327531, 0.0133938855,
      0.0145124272, 0.0397952050, 0.0303722005, 0.0200208705,
      0.0258587729, 0.1106555462, 0.0375629663, 0.1904202551,
      0.1363223642, 0.1082039401, 0.1414361298, 0.0527773313,
      0.1853451431, 0.1678386182, 0.0726319477, 0.0480239950,
      0.0842103213, 0.0744752362, 0.0660325885, 0.0913975239,
      0.0633665547, 0.0365940593, 0.0552844591, 0.0196380578,
      0.0192072298, 0.0725669637, 0.0784936771, 0.0972098336,
      0.0850971416, 0.0543594323, 0.0089790877, 0.0488873236,
      0.0927936360, 0.0787618235, 0.0485094227, 0.0455279350,
      0.0217985772, 0.0177213382, 0.0073623671, 0.0892393216,
      0.0640176609, 0.0143332323, 0.0414126925, 0.0049108928))).resize(1, 4, 5, 6)

    val expectedGradInput = Tensor(Storage(Array(
      0.0881276652, 0.1133036539, 0.0465447567, 0.0817324147,
      0.1428715140, 0.1434069127, 0.0200936515, 0.0331588611,
      0.0082347533, 0.0705295727, 0.0047801938, 0.0730933174,
      0.1038630530, 0.0445579700, 0.1082007885, 0.0945380554,
      0.0038371012, 0.0894166604, 0.0414803959, 0.0664161965,
      0.0453640148, 0.1109020710, 0.0704725906, 0.0250988398,
      0.0871438459, 0.1248503551, 0.0490181670, 0.0355132632,
      0.0620754026, 0.1498213857, 0.0878395960, 0.0605145358,
      0.0812550783, 0.0761175826, 0.0340194665, 0.0082995314,
      0.0588069893, 0.0502056703, 0.0325408317, 0.0202549081,
      0.0365867950, 0.0422046259, 0.0242312048, 0.0262613539,
      0.0411917791, 0.0774480551, 0.0527627692, 0.0255139079,
      0.0250179768, 0.0409159921, 0.0184869338, 0.0181240868,
      0.0462631583, 0.0078506442, 0.0435227007, 0.0325958617,
      0.0636918023, 0.0672071651, 0.0621983670, 0.0620262437,
      0.0149440048, 0.0267253947, 0.0135939466, 0.0229117554,
      0.0130322864, 0.0178058017, 0.0024611575, 0.0097070178,
      0.0388641059, 0.0092233680, 0.0276591033, 0.0260190759,
      0.0289575662, 0.0190035459, 0.0238665510, 0.0026787771,
      0.0029024854, 0.0079590408, 0.0060744402, 0.0040041744,
      0.0051717549, 0.0221311096, 0.0075125932, 0.0380840525,
      0.0272644740, 0.0216407888, 0.0282872263, 0.0105554666,
      0.0370690301, 0.0335677229, 0.0072631948, 0.0048023998,
      0.0084210327, 0.0074475235, 0.0066032591, 0.0091397529,
      0.0063366555, 0.0036594060, 0.0055284458, 0.0019638059,
      0.0019207230, 0.0072566965, 0.0078493683, 0.0097209839,
      0.0085097142, 0.0054359431, 0.0008979088, 0.0048887325,
      0.0092793638, 0.0078761829, 0.0048509422, 0.0045527937,
      0.0021798578, 0.0017721339, 0.0007362367, 0.0089239320,
      0.0064017661, 0.0014333233, 0.0041412693, 0.0004910893
    ))).resize(1, 4, 5, 6)

    val diff = scale.backward(input, outDiff)

    diff.map(expectedGradInput, (a, b) => {
      assert(Math.abs(a - b) < 1e-6)
      a
    })
  }

  "scale zeroParameter" should "work" in {

    val scale = new Scale[Double](Array(1, 4, 1, 1))
    scale.parameters()._1(0).copy(Tensor(Storage(Array(0.4, 0.3, 0.2, 0.1)))) // weight
    scale.parameters()._1(1).copy(Tensor(Storage(Array(0.1, 0.01, 0.03, 0.04)))) // bias
    val output = scale.forward(input)
    val gradOutput = Tensor[Double](1, 4, 5, 6).randn()
    scale.backward(input, gradOutput)

    println(scale.parameters()._2(0))
    println(scale.parameters()._2(1))

    scale.zeroGradParameters()

    scale.parameters()._2(0).apply1(x => {
      assert(x == 0); x
    })

    scale.parameters()._2(1).apply1(x => {
      assert(x == 0); x
    })
  }

}

class ScaleSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val scale = Scale[Float](Array(1, 4, 1, 1)).setName("scale")
    val input = Tensor[Float](1, 4, 5, 6).apply1(_ => Random.nextFloat())
    runSerializationTest(scale, input)
  }
}
