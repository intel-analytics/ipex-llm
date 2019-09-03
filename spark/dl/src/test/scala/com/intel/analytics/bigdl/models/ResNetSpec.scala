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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.resnet.{Convolution, ResNet, ResNetMask}
import com.intel.analytics.bigdl.models.resnet.ResNet._
import com.intel.analytics.bigdl.nn.Graph.{apply => _, _}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.{Graph, _}
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.torch.{TH, TorchSpec}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, T, Table}
import org.apache.log4j.Logger
import com.intel.analytics.bigdl.numeric.NumericFloat

import scala.collection.{immutable, mutable}
import scala.math._
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class ResNetSpec extends TorchSpec {

  private val suffix = ".t7" + (new java.util.Random()).nextLong()

  "ResNet basicBlockFunc graph" should "be same with original one" in {
    val depth = 16
    ResNetTest.iChannels = 16

    RandomGenerator.RNG.setSeed(1000)
    val model = ResNetTest.basicBlock(16, 1)
    RandomGenerator.RNG.setSeed(1000)
    val input = Input()
    val output = ResNetTest.basicBlockFunc(16, 1, input)
    val graphModel = Graph(input, output)

    val inputData = Tensor(4, 16, 32, 32).rand()
    val gradients = Tensor(4, 16, 32, 32).rand()

    val output1 = model.forward(inputData)
    val output2 = graphModel.forward(inputData)

    output1 should be(output2)

    val gradInput1 = model.backward(inputData, gradients)
    val gradInput2 = graphModel.backward(inputData, gradients)

    gradInput1 should be(gradInput2)
  }

  "ResNet bottleneckFunc graph" should "be same with original one" in {
    val depth = 16
    ResNetTest.iChannels = 16

    RandomGenerator.RNG.setSeed(1000)
    val model = ResNetTest.bottleneck(16, 1)
    RandomGenerator.RNG.setSeed(1000)
    val input = Input()
    val output = ResNetTest.bottleneckFunc(16, 1, input)
    val graphModel = Graph(input, output)


    val inputData = Tensor(4, 16, 32, 32).rand()
    val gradients = Tensor(4, 64, 32, 32).rand()

    val output2 = graphModel.forward(inputData).toTensor[Float]
    val output1 = model.forward(inputData).toTensor[Float]

    output1.size() should be (output2.size())
    output1 should be(output2)

    val gradInput1 = model.backward(inputData, gradients)
    val gradInput2 = graphModel.backward(inputData, gradients)

    gradInput1 should be(gradInput2)
  }

  "ResNet-18 graph" should "be same with original one for ImageNet" in {
    val batchSize = 4
    val classNum = 1000
    val depth = 18
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1( e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())

    RNG.setSeed(1000)
    val model = ResNet(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataSet" -> DatasetType.ImageNet))
    RNG.setSeed(1000)
    val graphModel = ResNet.graph(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataset" -> DatasetType.ImageNet))
    var modelForwardTime = 0L
    var modelBackwardTime = 0L
    var graphForwardTime = 0L
    var graphBackwardTime = 0L

    var output1: Tensor[Float] = null
    var output2: Tensor[Float] = null
    var st = System.nanoTime()
    for (i <- 1 to 3) {
      output1 = model.forward(input).toTensor[Float]
    }
    modelForwardTime += System.nanoTime() - st
    st = System.nanoTime()
    for (i <- 1 to 3) {
      output2 = graphModel.forward(input).toTensor[Float]
    }
    graphForwardTime += System.nanoTime() - st
    output1 should be(output2)

    var gradInput1: Tensor[Float] = null
    var gradInput2: Tensor[Float] = null
    st = System.nanoTime()
    for (i <- 1 to 3) {
      gradInput1 = model.backward(input, gradOutput).toTensor[Float]
    }
    modelBackwardTime += System.nanoTime() - st
    st = System.nanoTime()
    for (i <- 1 to 3) {
      gradInput2 = graphModel.backward(input, gradOutput).toTensor[Float]
    }
    graphBackwardTime += System.nanoTime() - st
    gradInput1 should be(gradInput2)

    val (modelF, modelB) = model.getTimes().map(v => (v._2, v._3))
      .reduce((a, b) => (a._1 + b._1, a._2 + b._2))
    val (graphF, graphB) = graphModel.getTimes().map(v => (v._2, v._3))
      .reduce((a, b) => (a._1 + b._1, a._2 + b._2))
    modelForwardTime should be (modelF +- modelF / 100)
    modelBackwardTime should be (modelB +- modelB / 100)
    graphForwardTime should be (graphF +- graphF / 100)
    graphBackwardTime should be (graphB +- graphB / 100)
  }


  "ResNet-50 graph" should "be same with original one for ImageNet" in {
    val batchSize = 4
    val classNum = 1000
    val depth = 50
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1( e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())

    RNG.setSeed(1000)
    val model = ResNet(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataSet" -> DatasetType.ImageNet))
    RNG.setSeed(1000)
    val graphModel = ResNet.graph(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataset" -> DatasetType.ImageNet))

    var output1: Tensor[Float] = null
    var output2: Tensor[Float] = null
    for (i <- 1 to 3) {
      output1 = model.forward(input).toTensor[Float]
      output2 = graphModel.forward(input).toTensor[Float]
    }
    output1 should be(output2)

    var gradInput1: Tensor[Float] = null
    var gradInput2: Tensor[Float] = null
    for (i <- 1 to 3) {
      gradInput1 = model.backward(input, gradOutput).toTensor[Float]
      gradInput2 = graphModel.backward(input, gradOutput).toTensor[Float]
    }
    gradInput1 should be(gradInput2)
  }

  "ResNet-50 graph test for maskrcnn" should "be same with original one for ImageNet" in {
    val batchSize = 4
    val classNum = 1000
    val depth = 50

    val input = Tensor[Float](T(T(T(
      T(-0.7767,  0.6317, -0.4749, -0.0322,  0.3530,  0.5078, -0.4746,
      -0.9143, -0.5839, -0.7639, -0.7566,  0.4712,  0.4235,  0.5752,
      -0.1634,  0.8028,  0.9938,  0.5130),
      T(-0.5522, -0.3953, -0.6432,  0.6477,  0.1115,  0.9540, -0.1120,
        0.8957,  0.4890, -0.0216, -0.5149,  0.4007,  0.0554, -0.5056,
        0.5818, -0.1530, -0.9662, -0.5582),
      T( 0.9071,  0.4128, -0.6742,  0.7804,  0.0325, -0.9282,  0.2953,
        -0.3139, -0.3635,  0.0522, -0.9107,  0.0247,  0.8102,  0.1977,
        -0.1100,  0.4557, -0.0875, -0.3223),
      T( 0.2423,  0.1060,  0.3792, -0.2625,  0.8105,  0.6712, -0.3922,
        0.3452,  0.1479,  0.8466,  0.8356,  0.5181,  0.5550,  0.2358,
        -0.3243, -0.5659,  0.8907,  0.4231),
      T(-0.7685,  0.3147, -0.3097, -0.9093,  0.9595,  0.1096,  0.3736,
        -0.0159, -0.8504,  0.9210, -0.3458, -0.9795,  0.9032, -0.4289,
        -0.5352,  0.8282,  0.5336, -0.6682),
      T(-0.1213, -0.5514,  0.7870, -0.9005, -0.6440, -0.3978, -0.6214,
        0.8372, -0.5737, -0.2087,  0.2033, -0.1532,  0.0449, -0.1650,
        -0.9320,  0.8315, -0.3842,  0.2539),
      T( 0.6554,  0.3187, -0.8226, -0.0221,  0.1775,  0.4680,  0.6994,
        0.8224, -0.0305,  0.8871, -0.2191, -0.5002, -0.3587,  0.9505,
        0.5164,  0.3376, -0.4698, -0.5328),
      T( 0.0114,  0.1377, -0.8733,  0.7986, -0.4535, -0.3205, -0.6241,
        0.1068, -0.4637,  0.9112,  0.9523,  0.1867, -0.3752,  0.8863,
        0.7037,  0.9631, -0.7736, -0.0434),
      T(-0.1127, -0.2305, -0.0958,  0.1137,  0.9905, -0.9970, -0.8375,
        -0.0185, -0.5740, -0.0794, -0.7228, -0.9446,  0.1325, -0.2994,
        0.3110,  0.5334, -0.5462,  0.5110),
      T( 0.2916, -0.2653, -0.6460, -0.4069,  0.9849, -0.5793, -0.7416,
        -0.6563,  0.8254,  0.3635, -0.6093,  0.9981, -0.7733, -0.9730,
        -0.7100,  0.5639, -0.3732, -0.4033),
      T(-0.3129, -0.5945,  0.9584, -0.0106, -0.2766,  0.9373, -0.9282,
        -0.3918,  0.9733, -0.7420,  0.3773, -0.6727, -0.8201, -0.3722,
        -0.7562, -0.2968, -0.5367, -0.4305),
      T(-0.2961, -0.4345, -0.5159, -0.0144,  0.1544, -0.2457, -0.5120,
        0.7988, -0.7918,  0.8386,  0.2403, -0.2683, -0.8754,  0.1935,
        -0.8342,  0.6370, -0.0072, -0.8823),
      T( 0.9681,  0.1671,  0.3474, -0.0523,  0.8672, -0.4885, -0.6988,
        0.5712, -0.1696,  0.1617, -0.7825,  0.4129, -0.9790, -0.0797,
        -0.4111, -0.9050,  0.2801, -0.2433),
      T( 0.1774, -0.8559,  0.8280, -0.9829, -0.5653, -0.6220, -0.8178,
        0.2688, -0.3717,  0.4104,  0.2894,  0.9034, -0.2837, -0.3177,
        -0.9135, -0.1253,  0.9895, -0.6503),
      T(-0.7253,  0.6009,  0.4007,  0.7606, -0.6855, -0.3315,  0.9304,
        -0.6277, -0.6984, -0.3634, -0.9358, -0.3421,  0.0603,  0.2801,
        0.5909, -0.3867, -0.5207, -0.7688),
      T(-0.0321, -0.2112, -0.8398,  0.5565,  0.3371, -0.5375, -0.7672,
        -0.6157, -0.5239, -0.6714, -0.6553,  0.6924, -0.7855,  0.4226,
        -0.7187, -0.4101, -0.3472, -0.0584),
      T(-0.2044,  0.4110,  0.8324,  0.6120,  0.4533,  0.6108, -0.6608,
        -0.5954,  0.8388, -0.9698, -0.9351,  0.9077,  0.1128,  0.5135,
        -0.6854, -0.2061, -0.5238, -0.7464),
      T(-0.1079, -0.9259,  0.2884,  0.6215, -0.4900,  0.7215,  0.6500,
        -0.5528, -0.8456, -0.0365, -0.8448, -0.8938, -0.4781, -0.7864,
        -0.3979, -0.0827,  0.0444, -0.8634),
      T( 0.8236,  0.6572, -0.6729, -0.6449,  0.4326,  0.8709, -0.7140,
        -0.2135, -0.7751, -0.3826,  0.9946, -0.1486,  0.3780,  0.9314,
        -0.9485, -0.1591, -0.8688, -0.0983),
      T(-0.8893, -0.3720,  0.4920,  0.8714,  0.7849, -0.7259, -0.6393,
        -0.1955, -0.1407, -0.2615, -0.6777,  0.8844,  0.7554,  0.0641,
        0.0784, -0.6840,  0.2840,  0.3862)),

      T(T(-0.9937,  0.3502, -0.6925,  0.0561, -0.7676, -0.1139, -0.5730,
        -0.5764,  0.3123, -0.2556, -0.2695,  0.4109, -0.8322, -0.6467,
        0.5978,  0.9476, -0.4670, -0.7182),
        T( 0.5260,  0.9383, -0.2584, -0.8752,  0.1734,  0.4347, -0.7645,
          0.3581, -0.6520, -0.6786,  0.1355, -0.0343,  0.6823, -0.5295,
          0.2935,  0.3608, -0.9663, -0.2061),
        T( 0.0704, -0.5954,  0.8112, -0.9855,  0.5505, -0.2575, -0.8574,
          -0.5125,  0.5218, -0.0707,  0.3545,  0.9719,  0.6294,  0.4823,
          0.9419,  0.1360, -0.5028, -0.5442),
        T( 0.5673, -0.4125, -0.7590, -0.2002, -0.8950, -0.6035,  0.6170,
          -0.4307,  0.0138, -0.9219, -0.3028,  0.3626,  0.2716,  0.0446,
          -0.2369, -0.8103,  0.3075,  0.1161),
        T( 0.6011, -0.9288,  0.6588,  0.7678, -0.6523,  0.4671,  0.4665,
          0.3260, -0.1567,  0.4127,  0.6477, -0.7431, -0.8806, -0.4552,
          0.4460,  0.3941,  0.7595, -0.3875),
        T(-0.1693, -0.7935, -0.4724,  0.4883, -0.9939, -0.3892, -0.4679,
          -0.9639, -0.0532, -0.1428, -0.7899, -0.9388, -0.2362, -0.0707,
          -0.2912,  0.9040, -0.4531, -0.5671),
        T( 0.0792,  0.5894, -0.1122, -0.2089, -0.7086, -0.1887,  0.6459,
          0.3378,  0.1729,  0.2001,  0.0400, -0.0124, -0.3343,  0.4281,
          0.3754,  0.6791, -0.5427, -0.6742),
        T( 0.4227,  0.5577, -0.2264, -0.7497,  0.3070, -0.4958, -0.5074,
          -0.6450,  0.4470,  0.6203, -0.7634, -0.6605, -0.3368,  0.7697,
          0.8174,  0.4651, -0.6808,  0.9072),
        T(-0.4782,  0.8635, -0.0853,  0.9245, -0.0010,  0.2295, -0.7131,
          -0.9983,  0.3084, -0.2764, -0.2133,  0.1740,  0.5011,  0.1126,
          0.2721,  0.2074,  0.7081, -0.4855),
        T( 0.4820, -0.2106,  0.2040,  0.8071,  0.8781, -0.4614,  0.9404,
          -0.2102, -0.0916,  0.6917,  0.8065, -0.5095, -0.4690, -0.3421,
          -0.5879,  0.3090,  0.8344,  0.5046),
        T(-0.4972,  0.7216,  0.2555,  0.8156, -0.4604, -0.8077, -0.9627,
          -0.3414,  0.7549, -0.4390, -0.3953, -0.6092, -0.6111, -0.2135,
          0.9189,  0.7397, -0.6067,  0.0623),
        T( 0.3616, -0.1882, -0.9984,  0.8188, -0.6950, -0.2005,  0.5873,
          -0.3700,  0.8219, -0.9748, -0.7416, -0.8625,  0.6110, -0.0923,
          0.8967, -0.4247, -0.8535,  0.6746),
        T( 0.2306, -0.4474,  0.0199,  0.6481,  0.1255, -0.9280,  0.3170,
          -0.9868,  0.1984,  0.7814,  0.8609,  0.5046,  0.8515, -0.5180,
          -0.9686, -0.5878,  0.0335,  0.3313),
        T( 0.8816,  0.2404,  0.8682,  0.5169, -0.9847,  0.4374,  0.0860,
          0.1257, -0.6400, -0.9781, -0.0396, -0.9107,  0.4829, -0.2877,
          0.7342, -0.2745,  0.3880, -0.0482),
        T(-0.4289,  0.2515, -0.3266,  0.3737, -0.7529, -0.0818, -0.5356,
          0.5235,  0.1423, -0.6117,  0.7922, -0.4498,  0.6471, -0.5820,
          -0.8728,  0.1719,  0.6032, -0.8251),
        T(-0.7598,  0.0936, -0.9353,  0.0304, -0.4683, -0.0040, -0.4607,
          0.9384,  0.4726,  0.0254, -0.0558,  0.3633, -0.4330, -0.4274,
          0.5457, -0.9579, -0.4633,  0.5689),
        T(-0.4948,  0.4971,  0.9163, -0.4537,  0.3993,  0.3803, -0.0650,
          0.6125, -0.8812,  0.1193, -0.4435, -0.7883,  0.2975, -0.9530,
          0.2860,  0.1074,  0.4756,  0.1964),
        T( 0.7179, -0.2535,  0.2406, -0.7721,  0.9244, -0.7792, -0.6800,
          0.9394, -0.4375,  0.8106,  0.0641,  0.7547, -0.7419, -0.1694,
          0.2915,  0.2083, -0.3227, -0.5777),
        T(-0.1488,  0.4304, -0.7608, -0.6858,  0.1218,  0.4829,  0.1727,
          0.0651,  0.5626,  0.6677, -0.0780,  0.2218, -0.3532,  0.0508,
          -0.0424,  0.3566,  0.1651, -0.3209),
        T( 0.7442,  0.3283, -0.7820, -0.2872, -0.4265, -0.9898, -0.1019,
          0.0210, -0.2512, -0.7553, -0.5404, -0.4416, -0.8282,  0.9510,
          0.9048,  0.6925, -0.6111, -0.8080)),

      T(T(-0.1142, -0.4107, -0.6453,  0.8724,  0.7219, -0.0073, -0.1745,
        0.8256, -0.3289, -0.9148,  0.2852,  0.6095,  0.8345,  0.7262,
        0.0392,  0.5028,  0.0234,  0.3259),
        T( 0.5919, -0.5676,  0.6113,  0.4649,  0.1327,  0.0144, -0.8684,
          0.3168, -0.4125, -0.0498,  0.8921, -0.0492,  0.3497,  0.8809,
          -0.0372, -0.7265,  0.2528,  0.7129),
        T(-0.9994,  0.4783,  0.5157, -0.8340, -0.1011, -0.6990,  0.5381,
          0.7556, -0.9029, -0.3316,  0.7034, -0.0408, -0.3513, -0.6281,
          -0.3031,  0.5648,  0.1794, -0.3823),
        T(-0.3217, -0.4526, -0.5716, -0.6036,  0.4017,  0.1012, -0.2849,
          0.9514, -0.2937,  0.0786, -0.8871, -0.4434,  0.6719,  0.0854,
          0.3333,  0.9832,  0.7973, -0.4942),
        T(-0.3105,  0.6190, -0.7508, -0.5638, -0.0714,  0.0814, -0.1162,
          0.4304,  0.0629,  0.0081, -0.6835, -0.0988,  0.6334, -0.3581,
          -0.9346,  0.3838, -0.0600,  0.5654),
        T(-0.4122,  0.7901, -0.2936,  0.0183,  0.5621,  0.8370, -0.1283,
          0.7334, -0.4812, -0.4205,  0.9054, -0.3940,  0.6789,  0.3079,
          0.1417, -0.5909, -0.9128,  0.0226),
        T( 0.8650,  0.5991, -0.2269, -0.5849,  0.1043,  0.2968,  0.3378,
          -0.6644,  0.6750, -0.1875,  0.1500,  0.9874,  0.3183, -0.7000,
          -0.2164,  0.1765, -0.1500,  0.0310),
        T(-0.2243, -0.9448, -0.1693, -0.1902,  0.8027,  0.4152, -0.2710,
          -0.5756,  0.0895, -0.5407, -0.3931,  0.2350, -0.9535, -0.5144,
          -0.0238,  0.2563,  0.7137,  0.9349),
        T(-0.1144, -0.8418,  0.3079, -0.3106,  0.5570,  0.2740, -0.5752,
          0.9643, -0.1315, -0.5235,  0.3923, -0.0203, -0.8707,  0.0961,
          0.7751,  0.1721,  0.0757, -0.2718),
        T( 0.7361, -0.8820,  0.3903, -0.0598, -0.9468,  0.2083,  0.0836,
          0.1768, -0.7008, -0.6665, -0.1816, -0.1314,  0.8482, -0.3006,
          0.6226, -0.3566, -0.3518, -0.5070),
        T(-0.3029, -0.5946,  0.8577,  0.8934,  0.2062, -0.6446,  0.7078,
          -0.3365, -0.5339, -0.4022, -0.9784, -0.4170, -0.3787, -0.1049,
          0.4976,  0.2958, -0.5584, -0.2234),
        T(-0.6633,  0.2375, -0.7615, -0.4331, -0.6776,  0.9655,  0.5955,
          -0.1696,  0.0489,  0.2039, -0.1507,  0.4565, -0.1931,  0.6757,
          -0.3218, -0.4453,  0.5237,  0.5118),
        T( 0.2712,  0.4025,  0.4534,  0.1564, -0.8237, -0.7177, -0.6903,
          0.9113, -0.7242, -0.1201,  0.4225,  0.7558,  0.4654,  0.0269,
          -0.6523,  0.7215,  0.5759, -0.9191),
        T(-0.7278,  0.3445,  0.3975,  0.9021,  0.7082,  0.2123, -0.9800,
          -0.0595, -0.8521, -0.2043,  0.9772,  0.2687,  0.2799, -0.9653,
          -0.6214,  0.3094,  0.0445,  0.5934),
        T( 0.2104,  0.3228, -0.0701, -0.0629,  0.4457, -0.5006,  0.6682,
          0.9234,  0.6650,  0.2999,  0.3199, -0.5112, -0.5266,  0.9799,
          -0.7906, -0.7965,  0.9010, -0.4560),
        T( 0.7867, -0.5065,  0.2917,  0.3705, -0.2114,  0.7550,  0.2628,
          0.1663,  0.2781,  0.4573, -0.1049,  0.6505, -0.5029,  0.5609,
          0.0971,  0.3630, -0.3165,  0.8093),
        T( 0.2900,  0.8729, -0.6296, -0.0511, -0.3488, -0.4096,  0.1224,
          -0.8658,  0.7535, -0.3045, -0.0223, -0.1846,  0.9813, -0.7863,
          0.2759,  0.3717,  0.4554, -0.5247),
        T( 0.0567, -0.2388,  0.2436,  0.4092,  0.6151,  0.6023, -0.6828,
          -0.6908, -0.3991,  0.1047, -0.3844, -0.5358,  0.9821,  0.1690,
          -0.3726,  0.0491, -0.4564,  0.5474),
        T( 0.4244,  0.6524,  0.2153, -0.5842,  0.4735,  0.1675, -0.8107,
          -0.3436,  0.5013, -0.8525,  0.6745,  0.4886,  0.0334,  0.0776,
          -0.9487, -0.9951, -0.7563, -0.8719),
        T(-0.6183, -0.4649, -0.7113, -0.8485, -0.1538,  0.0715,  0.2242,
          0.0629, -0.7821,  0.7599, -0.8241, -0.4093,  0.1514, -0.7615,
          -0.2735, -0.2368, -0.1939, -0.9840)))))

//    val input = Tensor[Float](T(T(T(
//      T(1.1166e-01, 8.1584e-01, 2.6256e-01, 4.8388e-01, 6.7650e-01,
//      7.5391e-01, 2.6269e-01, 4.2840e-02, 2.0803e-01, 1.1804e-01,
//      1.2169e-01, 7.3560e-01, 7.1177e-01, 7.8758e-01, 4.1831e-01,
//      9.0141e-01, 9.9689e-01, 7.5651e-01),
//      T(2.2389e-01, 3.0235e-01, 1.7842e-01, 8.2384e-01, 5.5574e-01,
//        9.7702e-01, 4.4402e-01, 9.4785e-01, 7.4448e-01, 4.8920e-01,
//        2.4256e-01, 7.0034e-01, 5.2769e-01, 2.4718e-01, 7.9089e-01,
//        4.2349e-01, 1.6899e-02, 2.2089e-01),
//      T(9.5354e-01, 7.0640e-01, 1.6288e-01, 8.9020e-01, 5.1627e-01,
//        3.5885e-02, 6.4763e-01, 3.4303e-01, 3.1824e-01, 5.2609e-01,
//        4.4659e-02, 5.1235e-01, 9.0508e-01, 5.9886e-01, 4.4500e-01,
//        7.2783e-01, 4.5626e-01, 3.3887e-01),
//      T(6.2115e-01, 5.5302e-01, 6.8960e-01, 3.6874e-01, 9.0527e-01,
//        8.3558e-01, 3.0390e-01, 6.7262e-01, 5.7396e-01, 9.2329e-01,
//        9.1781e-01, 7.5903e-01, 7.7752e-01, 6.1788e-01, 3.3787e-01,
//        2.1703e-01, 9.4536e-01, 7.1156e-01),
//      T(1.1574e-01, 6.5736e-01, 3.4515e-01, 4.5344e-02, 9.7976e-01,
//        5.5479e-01, 6.8678e-01, 4.9204e-01, 7.4799e-02, 9.6049e-01,
//        3.2709e-01, 1.0255e-02, 9.5160e-01, 2.8553e-01, 2.3242e-01,
//        9.1409e-01, 7.6681e-01, 1.6592e-01),
//      T(4.3933e-01, 2.2428e-01, 8.9348e-01, 4.9744e-02, 1.7798e-01,
//        3.0110e-01, 1.8930e-01, 9.1860e-01, 2.1314e-01, 3.9566e-01,
//        6.0166e-01, 4.2341e-01, 5.2244e-01, 4.1750e-01, 3.4009e-02,
//        9.1574e-01, 3.0789e-01, 6.2695e-01),
//      T(8.2768e-01, 6.5935e-01, 8.8695e-02, 4.8896e-01, 5.8873e-01,
//        7.3401e-01, 8.4972e-01, 9.1118e-01, 4.8474e-01, 9.4356e-01,
//        3.9045e-01, 2.4992e-01, 3.2063e-01, 9.7527e-01, 7.5818e-01,
//        6.6880e-01, 2.6512e-01, 2.3362e-01),
//      T(5.0568e-01, 5.6883e-01, 6.3373e-02, 8.9930e-01, 2.7325e-01,
//        3.3975e-01, 1.8794e-01, 5.5339e-01, 2.6817e-01, 9.5558e-01,
//        9.7613e-01, 5.9337e-01, 3.1239e-01, 9.4314e-01, 8.5186e-01,
//        9.8153e-01, 1.1318e-01, 4.7830e-01),
//      T(4.4365e-01, 3.8474e-01, 4.5208e-01, 5.5685e-01, 9.9523e-01,
//        1.4892e-03, 8.1264e-02, 4.9074e-01, 2.1299e-01, 4.6032e-01,
//        1.3862e-01, 2.7686e-02, 5.6623e-01, 3.5030e-01, 6.5549e-01,
//        7.6671e-01, 2.2692e-01, 7.5551e-01),
//      T(6.4580e-01, 3.6734e-01, 1.7702e-01, 2.9656e-01, 9.9246e-01,
//        2.1034e-01, 1.2920e-01, 1.7187e-01, 9.1269e-01, 6.8177e-01,
//        1.9534e-01, 9.9907e-01, 1.1334e-01, 1.3520e-02, 1.4501e-01,
//        7.8195e-01, 3.1339e-01, 2.9834e-01),
//      T(3.4357e-01, 2.0277e-01, 9.7920e-01, 4.9468e-01, 3.6170e-01,
//        9.6867e-01, 3.5917e-02, 3.0412e-01, 9.8666e-01, 1.2902e-01,
//        6.8866e-01, 1.6367e-01, 8.9926e-02, 3.1390e-01, 1.2192e-01,
//        3.5160e-01, 2.3164e-01, 2.8473e-01),
//      T(3.5195e-01, 2.8276e-01, 2.4204e-01, 4.9278e-01, 5.7719e-01,
//        3.7713e-01, 2.4401e-01, 8.9939e-01, 1.0409e-01, 9.1928e-01,
//        6.2014e-01, 3.6585e-01, 6.2283e-02, 5.9673e-01, 8.2883e-02,
//        8.1852e-01, 4.9640e-01, 5.8851e-02),
//      T(9.8403e-01, 5.8356e-01, 6.7370e-01, 4.7383e-01, 9.3360e-01,
//        2.5575e-01, 1.5061e-01, 7.8561e-01, 4.1519e-01, 5.8086e-01,
//        1.0876e-01, 7.0646e-01, 1.0476e-02, 4.6017e-01, 2.9446e-01,
//        4.7520e-02, 6.4007e-01, 3.7836e-01),
//      T(5.8870e-01, 7.2030e-02, 9.1402e-01, 8.5274e-03, 2.1736e-01,
//        1.8899e-01, 9.1108e-02, 6.3440e-01, 3.1417e-01, 7.0519e-01,
//        6.4472e-01, 9.5170e-01, 3.5815e-01, 3.4113e-01, 4.3272e-02,
//        4.3734e-01, 9.9474e-01, 1.7484e-01),
//      T(1.3736e-01, 8.0047e-01, 7.0036e-01, 8.8032e-01, 1.5727e-01,
//        3.3425e-01, 9.6522e-01, 1.8616e-01, 1.5079e-01, 3.1830e-01,
//        3.2107e-02, 3.2897e-01, 5.3014e-01, 6.4006e-01, 7.9543e-01,
//        3.0665e-01, 2.3967e-01, 1.1560e-01),
//      T(4.8393e-01, 3.9441e-01, 8.0108e-02, 7.7824e-01, 6.6856e-01,
//        2.3124e-01, 1.1641e-01, 1.9213e-01, 2.3805e-01, 1.6431e-01,
//        1.7237e-01, 8.4618e-01, 1.0725e-01, 7.1131e-01, 1.4063e-01,
//        2.9495e-01, 3.2638e-01, 4.7080e-01),
//      T(3.9782e-01, 7.0549e-01, 9.1618e-01, 8.0602e-01, 7.2666e-01,
//        8.0540e-01, 1.6962e-01, 2.0228e-01, 9.1942e-01, 1.5091e-02,
//        3.2429e-02, 9.5385e-01, 5.5642e-01, 7.5675e-01, 1.5729e-01,
//        3.9693e-01, 2.3810e-01, 1.2681e-01),
//      T(4.4604e-01, 3.7044e-02, 6.4422e-01, 8.1077e-01, 2.5499e-01,
//        8.6077e-01, 8.2499e-01, 2.2358e-01, 7.7200e-02, 4.8176e-01,
//        7.7594e-02, 5.3087e-02, 2.6097e-01, 1.0680e-01, 3.0105e-01,
//        4.5867e-01, 5.2221e-01, 6.8301e-02),
//      T(9.1178e-01, 8.2859e-01, 1.6354e-01, 1.7753e-01, 7.1628e-01,
//        9.3546e-01, 1.4298e-01, 3.9327e-01, 1.1245e-01, 3.0870e-01,
//        9.9732e-01, 4.2572e-01, 6.8901e-01, 9.6570e-01, 2.5725e-02,
//        4.2046e-01, 6.5620e-02, 4.5084e-01),
//      T(5.5344e-02, 3.1398e-01, 7.4600e-01, 9.3570e-01, 8.9247e-01,
//        1.3704e-01, 1.8033e-01, 4.0226e-01, 4.2963e-01, 3.6923e-01,
//        1.6114e-01, 9.4218e-01, 8.7770e-01, 5.3205e-01, 5.3919e-01,
//        1.5801e-01, 6.4201e-01, 6.9311e-01)),
//      T(T(3.1341e-03, 6.7510e-01, 1.5374e-01, 5.2807e-01, 1.1619e-01,
//        4.4307e-01, 2.1351e-01, 2.1179e-01, 6.5613e-01, 3.7220e-01,
//        3.6525e-01, 7.0546e-01, 8.3900e-02, 1.7666e-01, 7.9892e-01,
//        9.7380e-01, 2.6652e-01, 1.4092e-01),
//        T(7.6301e-01, 9.6913e-01, 3.7080e-01, 6.2380e-02, 5.8672e-01,
//          7.1736e-01, 1.1775e-01, 6.7905e-01, 1.7401e-01, 1.6068e-01,
//          5.6776e-01, 4.8285e-01, 8.4117e-01, 2.3526e-01, 6.4673e-01,
//          6.8039e-01, 1.6834e-02, 3.9697e-01),
//        T(5.3521e-01, 2.0231e-01, 9.0562e-01, 7.2718e-03, 7.7523e-01,
//          3.7126e-01, 7.1303e-02, 2.4375e-01, 7.6089e-01, 4.6466e-01,
//          6.7724e-01, 9.8594e-01, 8.1472e-01, 7.4113e-01, 9.7097e-01,
//          5.6798e-01, 2.4862e-01, 2.2790e-01),
//        T(7.8366e-01, 2.9373e-01, 1.2051e-01, 3.9992e-01, 5.2518e-02,
//          1.9826e-01, 8.0848e-01, 2.8465e-01, 5.0688e-01, 3.9037e-02,
//          3.4859e-01, 6.8130e-01, 6.3578e-01, 5.2229e-01, 3.8153e-01,
//          9.4870e-02, 6.5376e-01, 5.5807e-01),
//        T(8.0054e-01, 3.5621e-02, 8.2942e-01, 8.8390e-01, 1.7383e-01,
//          7.3356e-01, 7.3324e-01, 6.6301e-01, 4.2164e-01, 7.0633e-01,
//          8.2384e-01, 1.2846e-01, 5.9718e-02, 2.7242e-01, 7.2300e-01,
//          6.9703e-01, 8.7974e-01, 3.0623e-01),
//        T(4.1536e-01, 1.0325e-01, 2.6381e-01, 7.4417e-01, 3.0510e-03,
//          3.0540e-01, 2.6606e-01, 1.8062e-02, 4.7339e-01, 4.2858e-01,
//          1.0506e-01, 3.0599e-02, 3.8192e-01, 4.6466e-01, 3.5439e-01,
//          9.5199e-01, 2.7343e-01, 2.1644e-01),
//        T(5.3959e-01, 7.9468e-01, 4.4392e-01, 3.9554e-01, 1.4572e-01,
//          4.0563e-01, 8.2293e-01, 6.6888e-01, 5.8645e-01, 6.0006e-01,
//          5.2002e-01, 4.9382e-01, 3.3284e-01, 7.1403e-01, 6.8769e-01,
//          8.3954e-01, 2.2863e-01, 1.6288e-01),
//        T(7.1136e-01, 7.7883e-01, 3.8680e-01, 1.2513e-01, 6.5349e-01,
//          2.5209e-01, 2.4631e-01, 1.7751e-01, 7.2350e-01, 8.1017e-01,
//          1.1831e-01, 1.6977e-01, 3.3162e-01, 8.8487e-01, 9.0868e-01,
//          7.3256e-01, 1.5962e-01, 9.5361e-01),
//        T(2.6091e-01, 9.3173e-01, 4.5737e-01, 9.6227e-01, 4.9948e-01,
//          6.1475e-01, 1.4343e-01, 8.4358e-04, 6.5422e-01, 3.6181e-01,
//          3.9333e-01, 5.8699e-01, 7.5055e-01, 5.5630e-01, 6.3604e-01,
//          6.0368e-01, 8.5403e-01, 2.5727e-01),
//        T(7.4100e-01, 3.9469e-01, 6.0200e-01, 9.0354e-01, 9.3905e-01,
//          2.6928e-01, 9.7019e-01, 3.9490e-01, 4.5422e-01, 8.4587e-01,
//          9.0324e-01, 2.4525e-01, 2.6548e-01, 3.2897e-01, 2.0605e-01,
//          6.5448e-01, 9.1722e-01, 7.5231e-01),
//        T(2.5141e-01, 8.6081e-01, 6.2773e-01, 9.0779e-01, 2.6978e-01,
//          9.6172e-02, 1.8666e-02, 3.2928e-01, 8.7744e-01, 2.8049e-01,
//          3.0237e-01, 1.9541e-01, 1.9445e-01, 3.9323e-01, 9.5947e-01,
//          8.6987e-01, 1.9666e-01, 5.3115e-01),
//        T(6.8082e-01, 4.0589e-01, 8.1784e-04, 9.0942e-01, 1.5251e-01,
//          3.9976e-01, 7.9367e-01, 3.1498e-01, 9.1095e-01, 1.2593e-02,
//          1.2919e-01, 6.8727e-02, 8.0551e-01, 4.5387e-01, 9.4836e-01,
//          2.8766e-01, 7.3243e-02, 8.3731e-01),
//        T(6.1528e-01, 2.7631e-01, 5.0993e-01, 8.2403e-01, 5.6274e-01,
//          3.5981e-02, 6.5848e-01, 6.5825e-03, 5.9918e-01, 8.9070e-01,
//          9.3043e-01, 7.5230e-01, 9.2575e-01, 2.4099e-01, 1.5697e-02,
//          2.0611e-01, 5.1674e-01, 6.6563e-01),
//        T(9.4082e-01, 6.2020e-01, 9.3408e-01, 7.5845e-01, 7.6495e-03,
//          7.1872e-01, 5.4302e-01, 5.6283e-01, 1.8001e-01, 1.0971e-02,
//          4.8018e-01, 4.4650e-02, 7.4144e-01, 3.5617e-01, 8.6712e-01,
//          3.6274e-01, 6.9398e-01, 4.7592e-01),
//        T(2.8555e-01, 6.2576e-01, 3.3670e-01, 6.8683e-01, 1.2356e-01,
//          4.5910e-01, 2.3221e-01, 7.6177e-01, 5.7113e-01, 1.9417e-01,
//          8.9610e-01, 2.7509e-01, 8.2354e-01, 2.0900e-01, 6.3592e-02,
//          5.8594e-01, 8.0160e-01, 8.7466e-02),
//        T(1.2012e-01, 5.4681e-01, 3.2345e-02, 5.1519e-01, 2.6587e-01,
//          4.9799e-01, 2.6965e-01, 9.6922e-01, 7.3629e-01, 5.1270e-01,
//          4.7211e-01, 6.8166e-01, 2.8350e-01, 2.8628e-01, 7.7285e-01,
//          2.1026e-02, 2.6836e-01, 7.8446e-01),
//        T(2.5259e-01, 7.4857e-01, 9.5813e-01, 2.7315e-01, 6.9963e-01,
//          6.9015e-01, 4.6751e-01, 8.0624e-01, 5.9378e-02, 5.5964e-01,
//          2.7827e-01, 1.0583e-01, 6.4877e-01, 2.3487e-02, 6.4300e-01,
//          5.5371e-01, 7.3780e-01, 5.9821e-01),
//        T(8.5893e-01, 3.7326e-01, 6.2032e-01, 1.1397e-01, 9.6219e-01,
//          1.1038e-01, 1.6000e-01, 9.6972e-01, 2.8124e-01, 9.0531e-01,
//          5.3204e-01, 8.7737e-01, 1.2903e-01, 4.1530e-01, 6.4574e-01,
//          6.0413e-01, 3.3866e-01, 2.1114e-01),
//        T(4.2561e-01, 7.1518e-01, 1.1960e-01, 1.5712e-01, 5.6090e-01,
//          7.4147e-01, 5.8636e-01, 5.3256e-01, 7.8130e-01, 8.3386e-01,
//          4.6100e-01, 6.1088e-01, 3.2342e-01, 5.2542e-01, 4.7879e-01,
//          6.7830e-01, 5.8257e-01, 3.3956e-01),
//        T(8.7211e-01, 6.6414e-01, 1.0902e-01, 3.5640e-01, 2.8674e-01,
//          5.1134e-03, 4.4903e-01, 5.1049e-01, 3.7440e-01, 1.2236e-01,
//          2.2982e-01, 2.7918e-01, 8.5907e-02, 9.7549e-01, 9.5240e-01,
//          8.4624e-01, 1.9445e-01, 9.6013e-02)),
//
//      T(T(4.4292e-01, 2.9463e-01, 1.7736e-01, 9.3622e-01, 8.6094e-01,
//        4.9636e-01, 4.1277e-01, 9.1280e-01, 3.3556e-01, 4.2586e-02,
//        6.4261e-01, 8.0477e-01, 9.1726e-01, 8.6309e-01, 5.1961e-01,
//        7.5141e-01, 5.1170e-01, 6.6297e-01),
//        T(7.9596e-01, 2.1620e-01, 8.0565e-01, 7.3245e-01, 5.6633e-01,
//          5.0722e-01, 6.5816e-02, 6.5841e-01, 2.9376e-01, 4.7509e-01,
//          9.4606e-01, 4.7541e-01, 6.7484e-01, 9.4044e-01, 4.8139e-01,
//          1.3677e-01, 6.2642e-01, 8.5643e-01),
//        T(2.9749e-04, 7.3914e-01, 7.5787e-01, 8.2998e-02, 4.4943e-01,
//          1.5051e-01, 7.6906e-01, 8.7778e-01, 4.8540e-02, 3.3420e-01,
//          8.5171e-01, 4.7959e-01, 3.2434e-01, 1.8597e-01, 3.4846e-01,
//          7.8238e-01, 5.8968e-01, 3.0883e-01),
//        T(3.3913e-01, 2.7369e-01, 2.1418e-01, 1.9821e-01, 7.0086e-01,
//          5.5062e-01, 3.5757e-01, 9.7569e-01, 3.5315e-01, 5.3931e-01,
//          5.6438e-02, 2.7829e-01, 8.3593e-01, 5.4271e-01, 6.6664e-01,
//          9.9159e-01, 8.9863e-01, 2.5292e-01),
//        T(3.4477e-01, 8.0948e-01, 1.2461e-01, 2.1808e-01, 4.6428e-01,
//          5.4071e-01, 4.4189e-01, 7.1520e-01, 5.3144e-01, 5.0405e-01,
//          1.5826e-01, 4.5061e-01, 8.1672e-01, 3.2095e-01, 3.2688e-02,
//          6.9189e-01, 4.7002e-01, 7.8272e-01),
//        T(2.9391e-01, 8.9503e-01, 3.5319e-01, 5.0917e-01, 7.8104e-01,
//          9.1849e-01, 4.3584e-01, 8.6672e-01, 2.5941e-01, 2.8975e-01,
//          9.5270e-01, 3.0301e-01, 8.3946e-01, 6.5394e-01, 5.7087e-01,
//          2.0454e-01, 4.3592e-02, 5.1131e-01),
//        T(9.3249e-01, 7.9953e-01, 3.8654e-01, 2.0756e-01, 5.5216e-01,
//          6.4842e-01, 6.6889e-01, 1.6781e-01, 8.3752e-01, 4.0623e-01,
//          5.7501e-01, 9.9372e-01, 6.5916e-01, 1.5002e-01, 3.9179e-01,
//          5.8826e-01, 4.2498e-01, 5.1550e-01),
//        T(3.8784e-01, 2.7586e-02, 4.1537e-01, 4.0491e-01, 9.0137e-01,
//          7.0758e-01, 3.6451e-01, 2.1222e-01, 5.4475e-01, 2.2963e-01,
//          3.0346e-01, 6.1748e-01, 2.3228e-02, 2.4282e-01, 4.8810e-01,
//          6.2814e-01, 8.5683e-01, 9.6743e-01),
//        T(4.4282e-01, 7.9104e-02, 6.5393e-01, 3.4470e-01, 7.7849e-01,
//          6.3700e-01, 2.1242e-01, 9.8213e-01, 4.3427e-01, 2.3825e-01,
//          6.9616e-01, 4.8985e-01, 6.4634e-02, 5.4804e-01, 8.8754e-01,
//          5.8607e-01, 5.3786e-01, 3.6412e-01),
//        T(8.6804e-01, 5.8994e-02, 6.9515e-01, 4.7008e-01, 2.6604e-02,
//          6.0416e-01, 5.4179e-01, 5.8842e-01, 1.4958e-01, 1.6673e-01,
//          4.0918e-01, 4.3432e-01, 9.2412e-01, 3.4972e-01, 8.1131e-01,
//          3.2171e-01, 3.2411e-01, 2.4648e-01),
//        T(3.4854e-01, 2.0269e-01, 9.2885e-01, 9.4668e-01, 6.0312e-01,
//          1.7771e-01, 8.5388e-01, 3.3174e-01, 2.3307e-01, 2.9892e-01,
//          1.0792e-02, 2.9150e-01, 3.1064e-01, 4.4757e-01, 7.4882e-01,
//          6.4792e-01, 2.2079e-01, 3.8831e-01),
//        T(1.6836e-01, 6.1873e-01, 1.1926e-01, 2.8344e-01, 1.6121e-01,
//          9.8277e-01, 7.9773e-01, 4.1522e-01, 5.2445e-01, 6.0193e-01,
//          4.2463e-01, 7.2826e-01, 4.0345e-01, 8.3783e-01, 3.3908e-01,
//          2.7736e-01, 7.6185e-01, 7.5590e-01),
//        T(6.3559e-01, 7.0127e-01, 7.2668e-01, 5.7818e-01, 8.8163e-02,
//          1.4117e-01, 1.5483e-01, 9.5566e-01, 1.3790e-01, 4.3996e-01,
//          7.1127e-01, 8.7791e-01, 7.3269e-01, 5.1343e-01, 1.7383e-01,
//          8.6075e-01, 7.8794e-01, 4.0440e-02),
//        T(1.3609e-01, 6.7223e-01, 6.9875e-01, 9.5104e-01, 8.5412e-01,
//          6.0616e-01, 9.9973e-03, 4.7026e-01, 7.3939e-02, 3.9785e-01,
//          9.8861e-01, 6.3435e-01, 6.3997e-01, 1.7334e-02, 1.8929e-01,
//          6.5468e-01, 5.2227e-01, 7.9668e-01),
//        T(6.0519e-01, 6.6138e-01, 4.6496e-01, 4.6857e-01, 7.2285e-01,
//          2.4972e-01, 8.3411e-01, 9.6168e-01, 8.3252e-01, 6.4995e-01,
//          6.5993e-01, 2.4441e-01, 2.3671e-01, 9.8996e-01, 1.0470e-01,
//          1.0173e-01, 9.5052e-01, 2.7200e-01),
//        T(8.9336e-01, 2.4674e-01, 6.4587e-01, 6.8527e-01, 3.9431e-01,
//          8.7751e-01, 6.3141e-01, 5.8314e-01, 6.3905e-01, 7.2867e-01,
//          4.4753e-01, 8.2524e-01, 2.4854e-01, 7.8044e-01, 5.4854e-01,
//          6.8149e-01, 3.4173e-01, 9.0466e-01),
//        T(6.4500e-01, 9.3645e-01, 1.8519e-01, 4.7447e-01, 3.2560e-01,
//          2.9518e-01, 5.6118e-01, 6.7095e-02, 8.7673e-01, 3.4777e-01,
//          4.8885e-01, 4.0770e-01, 9.9065e-01, 1.0686e-01, 6.3795e-01,
//          6.8585e-01, 7.2769e-01, 2.3767e-01),
//        T(5.2833e-01, 3.8061e-01, 6.2182e-01, 7.0460e-01, 8.0757e-01,
//          8.0114e-01, 1.5859e-01, 1.5458e-01, 3.0045e-01, 5.5234e-01,
//          3.0778e-01, 2.3212e-01, 9.9107e-01, 5.8450e-01, 3.1371e-01,
//          5.2457e-01, 2.7181e-01, 7.7372e-01),
//        T(7.1221e-01, 8.2618e-01, 6.0763e-01, 2.0790e-01, 7.3675e-01,
//          5.8377e-01, 9.4661e-02, 3.2822e-01, 7.5067e-01, 7.3750e-02,
//          8.3726e-01, 7.4430e-01, 5.1672e-01, 5.3879e-01, 2.5634e-02,
//          2.4280e-03, 1.2183e-01, 6.4063e-02),
//        T(1.9086e-01, 2.6754e-01, 1.4437e-01, 7.5772e-02, 4.2309e-01,
//          5.3573e-01, 6.1210e-01, 5.3145e-01, 1.0893e-01, 8.7995e-01,
//          8.7953e-02, 2.9535e-01, 5.7568e-01, 1.1925e-01, 3.6326e-01,
//          3.8158e-01, 4.0306e-01, 7.9811e-03)))))

    val gradOutput = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())

    RNG.setSeed(1000)
//    val model = ResNetMask(classNum, T("shortcutType" -> ShortcutType.B,
//      "depth" -> depth, "dataSet" -> DatasetType.ImageNet))
//
//    model.evaluate()
////    val layer1 = ResNetMask.creatLayer1()
////    val layer2 = ResNetMask.creatLayer2()
////    val layer3 = ResNetMask.creatLayer3()
////    val layer4 = ResNetMask.creatLayer4()
////
////    layer1.evaluate()
////    layer2.evaluate()
////    layer3.evaluate()
////    layer4.evaluate()
//
//    val output1 = model.forward(input)
////    val out2 = layer1.forward(output1)
////    val out3 = layer2.forward(out2)
////    val out4 = layer3.forward(out3)
////    val out5 = layer4.forward(out4)
//
//    println("done")
  }

  "ResNet graph" should "be same with original one for Cifar10" in {
    val batchSize = 4
    val classNum = 10
    val depth = 20
    val input = Tensor[Float](32, 64, 128, 256).apply1( e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, classNum).apply1(e => Random.nextFloat())

    RNG.setSeed(1000)
    val model = ResNet(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataSet" -> DatasetType.CIFAR10))
    RNG.setSeed(1000)
    val graphModel = ResNet.graph(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataset" -> DatasetType.CIFAR10))

    var output1: Tensor[Float] = null
    var output2: Tensor[Float] = null
    for (i <- 1 to 3) {
      output1 = model.forward(input).toTensor[Float]
      output2 = graphModel.forward(input).toTensor[Float]
    }
    output1 should be(output2)

    var gradInput1: Tensor[Float] = null
    var gradInput2: Tensor[Float] = null
    for (i <- 1 to 3) {
      gradInput1 = model.backward(input, gradOutput).toTensor[Float]
      gradInput2 = graphModel.backward(input, gradOutput).toTensor[Float]
    }
    gradInput1 should be(gradInput2)
  }

}

object ResNetTest {
  val logger = Logger.getLogger(getClass)
  val opt = T()
  var iChannels = 0
  val depth = opt.get("depth").getOrElse(18)
  val shortCutType = opt.get("shortcutType")
  val shortcutType = shortCutType.getOrElse(ShortcutType.B).asInstanceOf[ShortcutType]
  val dataSet = opt.get("dataSet")
  val dataset = dataSet.getOrElse(DatasetType.CIFAR10).asInstanceOf[DatasetType]
  val optnet = opt.get("optnet").getOrElse(true)

  def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int): Module[Float] = {
    val useConv = shortcutType == ShortcutType.C ||
      (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

    if (useConv) {
      Sequential()
        .add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride, optnet = optnet))
        .add(SpatialBatchNormalization(nOutputPlane))
    } else if (nInputPlane != nOutputPlane) {
      Sequential()
        .add(SpatialAveragePooling(1, 1, stride, stride))
        .add(Concat(2)
          .add(Identity())
          .add(MulConstant(0f)))
    } else {
      Identity()
    }
  }


  def shortcutFunc(nInputPlane: Int, nOutputPlane: Int, stride: Int)(input: ModuleNode[Float])
  : ModuleNode[Float] = {
    val useConv = shortcutType == ShortcutType.C ||
      (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

    if (useConv) {
      val conv1 = Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride,
        optnet = optnet).inputs(input)
      val bn1 = SpatialBatchNormalization(nOutputPlane).inputs(conv1)
      bn1
    } else if (nInputPlane != nOutputPlane) {
      val pool1 = SpatialAveragePooling(1, 1, stride, stride).inputs(input)
      val mul1 = MulConstant(0f).inputs(pool1)
      val concat = JoinTable(2, 0).inputs(pool1, mul1)
      concat
    } else {
      input
    }
  }

  def basicBlock(n: Int, stride: Int): Module[Float] = {
    val nInputPlane = iChannels
    iChannels = n

    val s = Sequential()
    s.add(Convolution(nInputPlane, n, 3, 3, stride, stride, 1, 1, optnet = optnet))
    s.add(SpatialBatchNormalization(n))
    s.add(ReLU(true))
    s.add(Convolution(n, n, 3, 3, 1, 1, 1, 1, optnet = optnet))
    s.add(SpatialBatchNormalization(n))

    Sequential()
      .add(ConcatTable()
        .add(s)
        .add(shortcut(nInputPlane, n, stride)))
      .add(CAddTable(true))
      .add(ReLU(true))
  }

  def basicBlockFunc(n: Int, stride: Int, input: ModuleNode[Float])
  : ModuleNode[Float] = {
    val nInputPlane = iChannels
    iChannels = n

    val conv1 = Convolution(nInputPlane, n, 3, 3, stride, stride, 1, 1).inputs(input)
    val bn1 = SpatialBatchNormalization(n).inputs(conv1)
    val relu1 = ReLU(true).inputs(bn1)
    val conv2 = Convolution(n, n, 3, 3, 1, 1, 1, 1).inputs(relu1)
    val bn2 = SpatialBatchNormalization(n).inputs(conv2)
    val shortcut = shortcutFunc(nInputPlane, n, stride)(input)
    val add = CAddTable(true).inputs(bn2, shortcut)
    val output = ReLU(true).inputs(add)
    output
  }

  def bottleneck(n: Int, stride: Int): Module[Float] = {
    val nInputPlane = 16 // iChannels
    iChannels = n * 4

    val s = Sequential()
    s.add(Convolution(nInputPlane, n, 1, 1, 1, 1, 0, 0, optnet = optnet))
      .add(SpatialBatchNormalization(n))
      .add(ReLU(true))
      .add(Convolution(n, n, 3, 3, stride, stride, 1, 1, optnet = optnet))
      .add(SpatialBatchNormalization(n))
      .add(ReLU(true))
      .add(Convolution(n, n*4, 1, 1, 1, 1, 0, 0, optnet = optnet))
      .add(SpatialBatchNormalization(n * 4))

    Sequential()
      .add(ConcatTable()
        .add(s)
        .add(shortcut(nInputPlane, n*4, stride)))
      .add(CAddTable(true))
      .add(ReLU(true))
  }

  def bottleneckFunc(n: Int, stride: Int, input: ModuleNode[Float]): ModuleNode[Float] = {
    val nInputPlane = 16 // iChannels
    iChannels = n * 4

    val conv1 = Convolution(nInputPlane, n, 1, 1, 1, 1, 0, 0, optnet = optnet).inputs(input)
    val bn1 = SpatialBatchNormalization(n).inputs(conv1)
    val relu = ReLU(true).inputs(bn1)
    val conv2 = Convolution(n, n, 3, 3, stride, stride, 1, 1, optnet = optnet).inputs(relu)
    val bn2 = SpatialBatchNormalization(n).inputs(conv2)
    val relu2 = ReLU(true).inputs(bn2)
    val conv3 = Convolution(n, n*4, 1, 1, 1, 1, 0, 0, optnet = optnet).inputs(relu2)
    val sbn = SpatialBatchNormalization(n * 4).inputs(conv3)

    val shortcut = shortcutFunc(nInputPlane, n * 4, stride)(input)
    val add = CAddTable(true).inputs(sbn, shortcut)
    val output = ReLU(true).inputs(add)
    output
  }
}
