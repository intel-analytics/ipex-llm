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

import com.intel.analytics.bigdl.nn.mkldnn.Equivalent
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class FPNSpec extends FlatSpec with Matchers {
  "FPN updateOutput" should "work correctly" in {
    val in_channels_list = Array(1, 2, 4)
    val out_channels = 2
    val model = FPN[Float](in_channels_list, out_channels)

    val feature1 = Tensor(
      T(T(0.10110152, 0.10345000, 0.04320979, 0.84362656,
          0.59594363, 0.97288179, 0.34699517, 0.54275155),
        T(0.93956870, 0.07543808, 0.50965708, 0.26184946,
          0.92378283, 0.83272308, 0.54440099, 0.56682664),
        T(0.53608388, 0.74091697, 0.53824615, 0.12760854,
          0.70029002, 0.85137993, 0.01918983, 0.10134047),
        T(0.61024511, 0.11725241, 0.46950370, 0.15163177,
          0.99792290, 0.50036842, 0.65618765, 0.76569498),
        T(0.31238246, 0.96460360, 0.23587847, 0.94086981,
          0.15270233, 0.44916826, 0.53412461, 0.19992995),
        T(0.14841199, 0.95466810, 0.89249784, 0.10235202,
          0.24293590, 0.83814293, 0.78163254, 0.94990700),
        T(0.50397956, 0.23095572, 0.12026519, 0.70295823,
          0.80230796, 0.31913465, 0.86270124, 0.67926580),
        T(0.93120003, 0.08011329, 0.30662805, 0.97467756,
          0.32988423, 0.90689850, 0.46856666, 0.66390038)))
      .reshape(Array(1, 1, 8, 8))

    val feature2 = Tensor(
      T(T(T(0.30143285, 0.63111430, 0.45092928, 0.22753167),
          T(0.80318344, 0.67537767, 0.14698678, 0.45962620),
          T(0.21663177, 0.89086282, 0.92865956, 0.89360029),
          T(0.49615270, 0.46269470, 0.73047608, 0.12438315)),
        T(T(0.75820625, 0.59779423, 0.61585987, 0.35782731),
          T(0.36951083, 0.35381025, 0.64314663, 0.75517660),
          T(0.30200917, 0.69998586, 0.29572868, 0.46342885),
          T(0.41677684, 0.26154006, 0.16909349, 0.94081402))))
      .reshape(Array(1, 2, 4, 4))

    val feature3 = Tensor(
      T(T(T(0.57270211, 0.25789189),
          T(0.79134840, 0.62564188)),
        T(T(0.27365083, 0.43420678),
          T(0.61281836, 0.23570287)),
        T(T(0.21393263, 0.50206852),
          T(0.50650394, 0.73282623)),
        T(T(0.20319027, 0.06753725),
          T(0.18215942, 0.36703324))))
      .reshape(Array(1, 4, 2, 2))

    val inner1_w = Tensor(
      T(T(T(T(0.25616819)),
          T(T(-0.74193102)),
          T(T(0.22137421)),
          T(T(0.53996474))),
        T(T(T(-0.30102068)),
          T(T(0.24491900)),
          T(T(-0.84143710)),
          T(T(-0.73395455)))))
      .reshape(Array(1, 2, 4, 1, 1))
    val inner1_b = Tensor(T(0, 0))

    val inner2_w = Tensor(
      T(T(T(T(0.04691243)),
          T(T(-0.90420955))),
        T(T(T(1.09895408)),
          T(T(0.51624501)))))
      .reshape(Array(1, 2, 2, 1, 1))
    val inner2_b = Tensor(T(0, 0))

    val inner3_w = Tensor(
      T(T(T(T(0.24687862))),
        T(T(T(-0.56227243)))))
      .reshape(Array(1, 2, 1, 1, 1))
    val inner3_b = Tensor(T(0, 0))

    val layer1_w = Tensor(
      T(T(T(T(-0.04048228, 0.16222215, 0.10794550),
            T(-0.34169874, -0.25080314, 0.11539066),
            T(-0.27039635, 0.19380659, 0.19993830)),
          T(T(0.12585402, -0.38708800, 0.09077036),
            T(0.12301302, -0.29949811, 0.12835038),
            T(-0.32869643, 0.37100095, -0.26665413))),
        T(T(T(-0.23543328, -0.24697217, 0.15786803),
            T(0.19520867, -0.06484443, 0.39382762),
            T(-0.09158209, -0.22267270, 0.23828101)),
          T(T(0.16857922, -0.26403868, -0.07582438),
            T(0.31187642, -0.14743957, 0.19229126),
            T(-0.00750843, -0.21541777, -0.04269919)))))
      .reshape(Array(1, 2, 2, 3, 3))
    val layer1_b = Tensor(T(0, 0))

    val layer2_w = Tensor(
      T(T(T(T(-0.14214972, -0.17213514, -0.32127398),
            T(-0.23303765, -0.27284676, -0.05630624),
            T(-0.03209409, -0.16349350, -0.13884634)),
          T(T(0.05150193, -0.01451367, 0.29302871),
            T(0.38110715, 0.21102744, -0.01252702),
            T(-0.14486188, 0.39937240, 0.26671016))),
        T(T(T(-0.20462120, -0.03479487, -0.01640993),
            T(0.34504193, 0.11599201, 0.40438360),
            T(-0.17013551, 0.00606328, -0.14445123)),
          T(T(0.15805143, -0.06925225, -0.24366492),
            T(-0.16341771, -0.31556514, 0.03696010),
            T(0.07415351, -0.08760622, -0.17086124)))))
      .reshape(Array(1, 2, 2, 3, 3))
    val layer2_b = Tensor(T(0, 0))

    val layer3_w = Tensor(
      T(T(T(T(-0.21088375, 0.39961314, 0.28634924),
            T(-0.09605905, -0.09238201, 0.29169798),
            T(-0.16913360, 0.34432471, 0.12923980)),
          T(T(0.15992212, 0.11829317, -0.08958191),
            T(0.29556727, 0.28719366, 0.35837567),
            T(0.35775679, 0.13369364, 0.22401685))),
        T(T(T(0.23750001, -0.26816195, -0.33834153),
            T(0.02364820, -0.28069261, -0.31661153),
            T(-0.05442283, 0.30038035, 0.23050475)),
          T(T(0.24013102, -0.04941136, -0.01676598),
            T(0.36672127, -0.14019510, -0.18527937),
            T(-0.21643242, -0.06160817, 0.14386815)))))
      .reshape(Array(1, 2, 2, 3, 3))
    val layer3_b = Tensor(T(0, 0))

    val result1 = Tensor(
      T(T(T(T(-0.60857159, -0.49706429),
            T(-0.44821957, -0.69798434)),
          T(T(0.11003723, 0.24464746),
            T(0.21994369, -0.22257896)))))

    val result2 = Tensor(
      T(T(T(T(0.67646873, 0.75461042, 0.88370752, 0.72522950),
            T(0.80561060, 1.40666068, 0.81269693, 0.72721291),
            T(0.42856935, 0.57526082, 0.84400183, 0.24381584),
            T(0.60819602, 0.32838598, 0.17468216, -0.05505963)),
          T(T(-0.41587284, -0.59085888, -0.50279200, -0.25322908),
            T(-0.42020139, -0.64106256, -0.23952308, -0.29740968),
            T(-0.31366453, -0.12451494, -0.13788190, 0.07498236),
            T(-0.31522152, -0.13974780, -0.06333419, 0.15230046)))))

    val result3 = Tensor(
      T(T(T(T(-0.29643691, 0.32930288, 0.07719041, 0.20329267, -0.11702696,
              0.33030477, 0.19752777, 0.26074126),
            T(-0.04022884, -0.04050549, -0.17072679, 0.05824373, -0.18035993,
              -0.10781585, 0.21838233, 0.35475171),
            T(-0.14252800, -0.16825707, -0.28704056, -0.26278189, -0.19001812,
              0.20092483, 0.17245048, 0.46969670),
            T(-0.14943303, -0.45888224, 0.33286753, -0.42771903, 0.47255370,
              0.24915743, -0.21637592, 0.21200535),
            T(0.00808068, -0.16809230, -0.14534889, 0.29852685, 0.36068499,
              -0.19606119, -0.18463834, -0.19501874),
            T(-0.06999602, 0.55371714, -0.33532500, 0.29894528, 0.44789663,
              0.21802102, -0.32107252, -0.07110818),
            T(-0.19171244, 0.50532514, 0.00852559, -0.05432931, 0.56445789,
              -0.21175916, 0.01788443, 0.39967728),
            T(0.11412182, -0.05338766, 0.11950107, 0.33978215, 0.17466278,
              -0.22752701, 0.06036017, 0.51162905)),
          T(T(-0.18407047, -0.06274336, -0.19927005, -0.18067920, -0.12339569,
              -0.10210013, -0.13622473, 0.09764731),
            T(-0.21372095, -0.12506956, -0.10981269, -0.22901297, 0.15182146,
              0.01927174, -0.11695608, 0.25842062),
            T(-0.08454411, 0.00893094, 0.06784435, -0.36769092, 0.24231599,
              -0.07395025, -0.20645590, 0.32848105),
            T(0.07287200, 0.06812082, 0.00125982, -0.20824122, 0.26192454,
              -0.27801457, -0.43661070, 0.24346380),
            T(-0.08816936, -0.14699535, -0.50232911, 0.17301719, 0.39865568,
              0.21348065, 0.22505483, 0.28257197),
            T(0.12479763, -0.03339935, -0.48426947, 0.55722409, 0.36770806,
              -0.01681852, 0.11375013, 0.19888467),
            T(0.14368367, 0.01942967, -0.23314725, 0.41997516, 0.39273715,
              -0.40041974, -0.07516777, 0.04501504),
            T(-0.00356270, -0.15851222, 0.04203597, 0.33169088, -0.02303683,
              -0.42069232, -0.08245742, 0.06082898)))))

    val input = T(feature1, feature2, feature3)
    val expectedOutput = T(result1, result2, result3)

    model.parameters()._1(0).copy(inner1_w)
    model.parameters()._1(1).copy(inner1_b)
    model.parameters()._1(2).copy(inner2_w)
    model.parameters()._1(3).copy(inner2_b)
    model.parameters()._1(4).copy(inner3_w)
    model.parameters()._1(5).copy(inner3_b)
    model.parameters()._1(6).copy(layer1_w)
    model.parameters()._1(7).copy(layer1_b)
    model.parameters()._1(8).copy(layer2_w)
    model.parameters()._1(9).copy(layer2_b)
    model.parameters()._1(10).copy(layer3_w)
    model.parameters()._1(11).copy(layer3_b)

    val output = model.forward(input)

    Equivalent.nearequals(output.toTable.get[Tensor[Float]](1).get,
      expectedOutput.get[Tensor[Float]](1).get) should be(true)
    Equivalent.nearequals(output.toTable.get[Tensor[Float]](2).get,
      expectedOutput.get[Tensor[Float]](2).get) should be(true)
    Equivalent.nearequals(output.toTable.get[Tensor[Float]](3).get,
      expectedOutput.get[Tensor[Float]](3).get) should be(true)
  }
}

class FPNSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val input = T()
    val feature1 = Tensor[Float](1, 1, 8, 8).apply1(_ => Random.nextFloat())
    val feature2 = Tensor[Float](1, 2, 4, 4).apply1(_ => Random.nextFloat())
    val feature3 = Tensor[Float](1, 4, 2, 2).apply1(_ => Random.nextFloat())
    input(1.0f) = feature1
    input(2.0f) = feature2
    input(3.0f) = feature3

    val fpn = new FPN[Float](inChannels = Array(1, 2, 4), outChannels = 2).setName("FPN")
    runSerializationTest(fpn, input)
  }
}
