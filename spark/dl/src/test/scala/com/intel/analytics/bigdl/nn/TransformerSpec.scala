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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class TransformerLayerSpec extends FlatSpec with Matchers {
  "tranformer decode stack" should "work correctly" in {
    val vocabSize = 10
    val hiddenSize = 4
    val numHeads = 2
    val filterSize = 3
    val num_hidden_layers = 1
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val transformer = new Transformer[Float](vocabSize,
      hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    val input1 = Input[Float]()
    val input2 = Input[Float]()

    val blockOutput = transformer.block(num_hidden_layers, input1, input2, blockType = "encode")
    val block = Graph(Array(input1, input2), blockOutput)
    val paramsTable = block.getParametersTable()

    for (i <- paramsTable.keySet) {
      if (i.toString contains "_q") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.12254566, -0.3492695, 0.6760147, 0.4690166),
            T(-0.70616156, -0.7172935, -0.70902413, -0.7268282),
            T(-0.17867321, 0.03752673, 0.21406537, -0.84105927),
            T(-0.40054652, 0.01422167, 0.49654406, -0.62966037))).t())
      } else if (i.toString contains "_k") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.80201703, 0.29880065, 0.8191585, 0.393151),
            T(-0.43785518, 0.02502167, -0.85530514, 0.86387163),
            T( 0.07737422, 0.34640843, 0.5547114, 0.12658376),
            T( 0.6287202, -0.7140273, -0.08061278, -0.3983137))).t())
      } else if (i.toString contains "_v") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.14568096, 0.8488055, -0.38585222, -0.42583144),
            T(-0.35776895, 0.00440949, 0.76952034, 0.7039148),
            T(-0.4635923, -0.5273898, 0.36311775, 0.21081167),
            T(-0.04171634, 0.24859089, 0.03242427, -0.01675642))).t())
      } else if (i.toString contains "_output_transform") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T( 0.8254406, 0.7399195, -0.76593506, -0.38950253),
              T( 0.51289314, 0.1285783, -0.24543494, -0.7138509),
              T(-0.34158242, -0.37842813, -0.5111934, 0.5966528),
              T( 0.39076942, -0.7022542, 0.8254971, -0.50844))).t())
      } else if (i.toString contains "_filter_layer") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T( 0.4929167, -0.5465611, 0.4262464),
              T( 0.5161569, -0.6786176, 0.37465477),
              T( 0.35582626, 0.43647707, -0.23218763),
              T( 0.7624726, 0.28653884, 0.20991063))).transpose(1, 2))
      } else if (i.toString contains "_output_layer") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(-0.9037433, 0.6076299, 0.6593666, -0.06372046),
              T( 0.58014977, 0.6601094, -0.72481453, 0.89943814),
              T( 0.02975523, -0.4040287, 0.6437061, -0.2594086))).transpose(1, 2))
      }
    }

    val input = Tensor[Float](Tensor[Float](
      T(T(T( 2.43651805, -0.91763462, -0.79225763, -1.60945293),
        T( 1.29811144, -3.45230805, 2.61721765, -1.14181035),
        T( 0.47855864, -0.37405556, 2.19316191, -3.09021106),
        T(-0.48362581, -0.57608153, 1.70065416, -1.6498369),
        T(-0.25864231, -1.31678763, 0.06332062, 0.87422282),
        T(-1.65092877, 1.71708556, 1.35238608, 0.75374151)),
        T(T( 1.35128392, -1.02559179, -0.18433534, -1.40365415),
          T(-0.40183212, 0.7955332, -1.03749113, -0.59513029),
          T(-1.03075905, -1.26780846, -1.0068692, -0.0189969),
          T(-1.67596552, 0.35162355, 2.48970327, 1.11306624),
          T(-0.28775333, -1.33144345, -1.12073744, 2.5386819),
          T( 0.07621163, -0.95549347, 0.28637323, 3.1503827)))))
    val bias = Tensor[Float](Tensor[Float](
      T(T(T(T( 0.12015895, 0.61720311, 0.30017032, -0.35224985, -1.1425182, -0.34934272),
        T(-0.20889423, 0.58662319, 0.83898341, 0.93110208, 0.28558733, 0.88514116),
        T(-0.75439794, 1.25286816, 0.51292982, -0.29809284, 0.48851815, -0.07557171),
        T( 1.13162939, 1.51981682, 2.18557541, -1.39649634, -1.44411381, -0.50446586),
        T( 0.16003707, 0.87616892, 0.31563495, -2.02220122, -0.30620401, 0.82797464),
        T( 0.23009474, 0.76201118, -0.22232814, -0.20075807, 0.18656139, 0.41005165))))
    ))

    val expectedOutput = Tensor[Float](
      T(T(T( 1.6739436, -0.5742816, -0.18686886, -0.91279316),
        T( 0.56332755, -1.6895478, 0.8744801, 0.25174013),
        T( 0.18294929, -0.03678755, 1.333065, -1.4792268),
        T(-0.83871794, 0.09105678, 1.6003608, -0.8526995),
        T(-0.6227458, 0.06268612, -1.0336334, 1.593693),
        T(-1.6069404, 0.70157117, -0.05510008, 0.9604694)),
        T(T( 1.500092, -0.12251449, -0.06195105, -1.3156266),
          T( 0.88058877, 0.88686943, -0.2218959, -1.5455623),
          T(-1.73186, 0.59709984, 0.5559552, 0.5788053),
          T(-1.7018749, 0.8331325, 0.30757982, 0.56116235),
          T(-0.5026365, -0.1983719, -0.96522677, 1.6662351),
          T(-0.56770575, -0.17644365, -0.92594254, 1.6700919)))
    )

    val output = block.forward(T(input, bias))
    output should be(expectedOutput)

    val gradInput = block.backward(T(input, bias), output).toTable

    val expectedGradInput1 = Tensor[Float](
      T(T(T( 9.1339905e-07, -4.4728981e-07, -3.1617617e-07, -1.2013072e-07),
        T( 6.3113339e-07, -5.3439135e-07, -3.3880960e-07, 2.1226521e-07),
        T( 1.7116510e-06, -3.8029987e-07, -5.2847190e-07, -7.6935152e-07),
        T(-1.0739063e-06, 3.4577083e-06, 5.1119628e-06, -7.4957657e-06),
        T(-3.0945554e-07, -2.9928319e-07, -2.0712267e-07, 8.1958660e-07),
        T(-1.0444269e-06, 1.6913917e-07, 3.5171459e-07, 4.8632023e-07)),
        T(T(-5.9887736e-07, 1.0681491e-06, 3.0668511e-06, -3.5249473e-06),
        T( 8.1442304e-06, 1.8526016e-05, 7.9063993e-06, -3.4815068e-05),
        T(-3.3653050e-05, 6.2354911e-07, 1.1947766e-05, 2.1081711e-05),
        T( 8.8148295e-07, -1.5203238e-06, 4.0907385e-06, -3.4518978e-06),
        T(-1.8306933e-06, 1.3375227e-06, -1.5494516e-06, 1.9532151e-06),
        T(1.1980649e-06, 9.9704266e-07, -3.0255887e-06, 8.8263494e-07)))
    )

    val expectedGradInput2 = Tensor[Float](
      T(T(T(T(1.92614536e-07, 8.18386638e-08, 1.83079862e-07, -5.29573754e-07,
        2.14264446e-07, -1.42223712e-07),
        T( 9.00455632e-07, -4.55583267e-06, 4.20768583e-06, -8.96842812e-06,
          5.02361490e-06, 3.39250482e-06),
        T(-3.51306221e-06, 1.35622076e-06, -2.57200622e-06, 1.08205404e-05,
          -4.62260732e-06, -1.46908474e-06),
        T(1.44854653e-06, 1.00405509e-06, -1.88945739e-06, -8.24743935e-08,
          1.16377095e-07, -5.97047006e-07),
        T(-5.35773211e-07, 1.24227370e-07, 1.73641411e-07, 1.35646133e-07,
          -1.13603612e-07, 2.15861803e-07),
        T(-6.30813645e-07, 6.52564225e-08, 1.47730645e-07, 3.11057221e-07,
          7.64788126e-08, 3.02906109e-08))))
    )

    val expectedGradWeights = Tensor[Float](
      T(-9.4019833e-06, -1.5453285e-06, -9.4909010e-06, 5.2547603e-07,
      4.0047512e-06, -7.0249803e-06, 1.3278475e-05, 4.4464814e-06,
      -3.8477524e-06, -8.1469705e-07, -1.3136616e-06, 1.5246084e-06,
        -3.6292959e-06, -1.3310753e-05, 1.0742175e-05, -1.3015128e-05,
        -2.8296442e-06, 4.6112955e-06, -2.7704493e-06, 6.8603067e-06,
        1.0306692e-05, 9.5141559e-06, -6.6580633e-06, 4.6302143e-06,
      -5.6733006e-06, -2.0463798e-05, -2.8769139e-06, -9.0087809e-07,
        4.6731147e-06, 1.3545281e-05, 1.2833587e-05, 1.6316061e-06,
        6.7491310e-06, 1.9667668e-05, 1.9997810e-07, 2.7255970e-07,
        -5.7489456e-06, -1.2749153e-05, -1.0156651e-05, -1.0032876e-06,
      -6.4571459e-06, -4.5748075e-06, 5.2935420e-06, 1.7019968e-06,
        -2.8230164e-05, -2.3874696e-05, -3.1409923e-05, 1.4136816e-05,
        1.3851404e-05, 1.3069550e-05, 3.1755158e-06, 4.1450826e-06,
        2.0835905e-05, 1.5379959e-05, 2.2940867e-05, -1.9983896e-05,
      3.9784982e-07, 2.6731566e-06, -5.9224215e-07, -2.5417473e-06,
        -7.5010930e-06, 1.6819112e-06, 1.4458296e-06, 4.3033779e-06,
        3.2009964e-05, 7.8872072e-06, -1.2185321e-05, -2.7866208e-05,
        3.1262254e-05, 8.2735351e-06, -1.2112221e-05, -2.7589167e-05,
      -1.0220035e-05, 8.3456416e-06, -3.1375414e-05, 4.9415255e-05,
      2.3259896e-05, 2.5363222e-05, 1.2638515e-05, 2.9357281e-05,
      -1.6661495e-05, 4.0918521e-06, 5.3757998e-07,
        1.8960893e-05, 8.4753447e-07, -2.3114646e-06,
        -6.3083702e-05, -1.7887363e-05, 4.8286256e-06,
        6.0784321e-05, 1.2947977e-05, -3.0547415e-06,
      4.2135795e-05, -7.1919526e-06, -3.3792276e-06,
      -2.0067891e-05, 6.7602373e-06, 5.6371910e-06, 7.6476235e-06,
        -6.6570569e-06, 1.3790517e-06, 5.3389158e-06, -9.7211682e-08,
        -1.1374552e-05, 2.0630792e-05, 4.2232737e-06, -1.3708518e-05,
      -2.71759927e-05, 2.76453793e-05, 1.34781003e-05, -1.42119825e-05,
      16.322777, 5.6128163, 8.455086, 17.609234,
      -2.7715797, 0.37446928, 1.2208222, 1.176289)
    )

    require(gradInput[Tensor[Float]](1).almostEqual(expectedGradInput1, 1e-6) == true)
    require(gradInput[Tensor[Float]](2).almostEqual(expectedGradInput2, 1e-6) == true)
  }

  "tranformer for translation" should "work correctly" in {
    val vocabSize = 16
    val hiddenSize = 4
    val filterSize = 8
    val numHeads = 1
    val num_hidden_layers = 1
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val transformer = new Transformer[Float](vocabSize,
      hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout, withShareWeightsLinear = true,
      transformerType = Translation)

    val attention0 = transformer.model("encode_self_attention_0/self_attention").get
    val ffn0 = transformer.model("encode_ffn_0/ffn").get

    val attention1 = transformer.model("decode_self_attention_0/self_attention").get
    val ffn1 = transformer.model("decode_ffn_0/ffn").get
    val attention2 = transformer.model("decode_encdec_attention_0/encdec_attention").get

    var paramsTable = attention0.getParametersTable()
    for (i <- paramsTable.keySet) {
      if (i.toString contains "_q") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(0.6719899, 0.29741684, -0.6073703, 0.58373296),
            T(0.28487056, 0.12325107, -0.18469666, -0.3146433),
            T(0.60392314, 0.65988046, 0.50996345, -0.19420744),
            T(0.40057203, -0.9149872, 0.10390836, 0.97260743))).t())
      } else if (i.toString contains "_k") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(0.33549386, 0.88536686, -0.30634838, 0.05587747),
            T(0.61110026, -0.66457653, -0.34049615, -0.14537863),
            T(0.653832, 0.74835855, 0.76725274, -0.6947307),
            T(0.49148628, -0.07944908, -0.845008, 0.6068878))).t())
      } else if (i.toString contains "_v") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(0.24006118, 0.6792713, 0.22704636, 0.49668023),
            T(0.53909445, -0.32836607, 0.25972122, 0.5554116),
            T(-0.4319324, 0.43911168, 0.20273127, -0.24734582),
            T(0.23329619, -0.3165343, 0.40053207, -0.34865358))).t())
      } else if (i.toString contains "_output_transform") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(-0.5211139, -0.3813012, 0.34638476, -0.21196833),
              T(0.1121366, -0.3850857, 0.15838127, -0.46018872),
              T(0.42922392, 0.49836066, -0.00889128, -0.20409666),
              T(-0.0800805, 0.6680052, 0.11346864, -0.3564058))).t())
      }
    }

    paramsTable = ffn0.getParametersTable()
    for (i <- paramsTable.keySet) {
      if (i.toString contains "_filter_layer") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(-0.42055795, -0.345141, -0.77015144, 1.0128733, -0.2070824,
              0.41457736, -0.27325338, 0.37545303),
              T(0.83861953, 0.49639514, 0.10912374, 0.4054078, 0.01117581,
                0.4838021, 0.47710165, 0.23820893),
              T(-0.37739983, -0.3799013, 0.26106557, -0.02527841, -0.09814293,
                0.15995328, 0.76590466, -0.38680843),
              T(0.22057502, 0.4438025, 0.18568423, 0.2206358, -0.5293094,
                -0.07671213, -0.5392774, -0.26026365))).transpose(1, 2))
      } else if (i.toString contains "_output_layer") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(-0.15800391, 0.00911217, 0.5716306, -0.4307602),
              T(-0.17119521, 0.45397595, -0.15994692, 0.1173245),
              T(0.02792565, 0.1785465, 0.03194377, -0.2635249),
              T(-0.5619625, 0.34994912, 0.2134058, 0.17008546),
              T(-0.16928878, -0.04155388, -0.00634552, 0.10220164),
              T(-0.19378763, 0.60514146, 0.31211597, 0.32819757),
              T(-0.12504072, -0.5004057, -0.53571004, -0.6392757),
              T(-0.06203287, 0.25287995, 0.32892716, 0.11961207))).transpose(1, 2))
      }
    }

    paramsTable = attention1.getParametersTable()
    for (i <- paramsTable.keySet) {
      if (i.toString contains "_q") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.58024985, -0.48674917, -0.1278461, -0.1681186),
            T(1.0511181, 0.50676775, -0.49831128, -0.13611957),
            T(0.4512829, 0.00988893, 0.35473365, -0.4541598),
            T(-0.01564673, -0.06611676, 0.20534483, -0.13249157))).t())
      } else if (i.toString contains "_k") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(0.25792515, 0.8091696, -1.1157143, -0.48759258),
            T(0.2797681, -0.61634296, 0.29310933, 0.3868902),
            T(-0.22521666, -0.08918925, 0.17066494, 0.06447314),
            T(-0.14935619, -0.05546288, -1.134581, 0.33467665))).t())
      } else if (i.toString contains "_v") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.05646669, 0.2533887, 0.9146523, 0.09979013),
            T(-0.03409033, 0.9656157, -0.00790233, 0.22394712),
            T(0.44499645, -0.41030893, -0.40253338, -0.541713),
            T(0.63082635, 0.05910337, 0.26689664, 0.06098993))).t())
      } else if (i.toString contains "_output_transform") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(0.07528905, -0.6294302, -0.47716418, -0.3372765),
              T(-0.4738406, -0.09567301, -0.21502851, 0.07263356),
              T(0.21500742, -0.09957578, 0.05073479, 0.5063499),
              T(-0.95140356, -0.19597691, 0.3108005, 0.3067237))).t())
      }
    }

    paramsTable = attention2.getParametersTable()
    for (i <- paramsTable.keySet) {
      if (i.toString contains "_q") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.09555588, 0.16374706, -0.81079763, 0.18353464),
            T(0.72976017, -0.6785369, -0.1633139, -0.1220759),
            T(-0.47357813, 0.19808318, 0.63312566, -0.14370666),
            T( 0.11398887, 0.7884044, -0.36504376, -0.17514746))).t())
      } else if (i.toString contains "_k") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.19676681, -0.24631989, -1.1253904, -0.2751462),
            T(-0.17718858, 0.06754616, 0.5731753, -0.8507766),
            T( 0.06555229, -0.04867446, -0.05025194, -0.5535116),
            T(-0.5346166, 0.23926297, -0.4628236, -0.3947385))).t())
      } else if (i.toString contains "_v") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T( 0.92687607, -0.545517, -0.05255984, 0.28678837),
            T( 0.34195843, 0.3929567, 0.51847, 0.7892322),
            T( 0.90397906, -0.9298378, 0.8783962, 0.2852646),
            T( 0.6237778, 0.3783044, 0.37894192, 0.42552295))).t())
      } else if (i.toString contains "_output_transform") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(-1.9982174e-01, 1.4843611e-01, 4.4536388e-01, -3.4881935e-01),
              T(6.5677509e-02, 7.3198605e-01, 4.1394565e-01, 3.6246496e-01),
              T(3.8297844e-01, -2.0218496e-01, -6.0479283e-01, -8.4035518e-04),
              T(8.8539845e-01, 8.1015706e-02, -2.0919992e-01, -3.2815292e-01))).t())
      }
    }
    paramsTable = ffn1.getParametersTable()
    for (i <- paramsTable.keySet) {
      if (i.toString contains "_filter_layer") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(-0.3522124, -0.51549995, -0.67411846, 0.27011815, 0.6126283, -0.5052634,
               0.88756555, 0.47037336),
            T( 0.15704805, -0.11248052, 0.45173776, 1.0609885, -0.02032901, -0.272949,
              -0.27566594, 0.45384774),
            T( 0.6470523, -0.6543102, -0.21736439, -0.43480754, -0.13311917, -1.1141537,
              -0.59988606, -0.24346256),
            T( 0.11163724, -0.03015788, 0.38666677, -0.39999688, -0.53780854, -0.09386043,
              -0.09019023, 0.28964663))).transpose(1, 2))
      } else if (i.toString contains "_output_layer") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(-0.28514335, -0.5174819, -0.3048153, 0.16713372),
              T(-0.2276286, -0.31804547, 0.269992, 0.03182783),
              T(-0.26096576, -0.49425197, -0.23944728, 0.28338984),
              T( 0.260591, -0.17206982, -0.14490226, -0.20425473),
              T( 0.38700444, -0.5851576, 0.309289, -0.28129402),
              T(-0.03296154, -0.47809625, 0.43516076, 0.21953852),
              T(-0.38866428, 0.52283365, -0.60793763, 0.33401495),
              T(-0.29918984, 0.6243824, -0.21915461, -0.14608558))).transpose(1, 2))
      }
    }

    val expectedOutput = Tensor[Float](
      T(T(T(1.0213897, 1.6298342, -0.77117467, -0.30903974,
        1.079551, -2.0169363, 0.5122813, -0.28446424,
        1.6982273, 0.98818946, 0.9912475, 0.3734624,
        -0.07386526, 0.7457521, 0.7346176, 0.5543957),
        T(1.1742185, 1.6519041, -0.74003303, -0.36850277,
          1.3430122, -2.163493, 0.6831105, -0.21015275,
          1.56321, 0.9189906, 1.0753409, 0.06065345,
          -0.08118278, 1.0861892, 0.40153468, 0.5656462),
        T(-1.641943, -0.54821557, -0.10801831, 1.3602101,
          -1.0806575, 1.2603211, -0.95736504, -0.97358,
          -0.3041229, -1.0635418, 0.9337779, -0.92391706,
          0.51814425, -0.49280763, 1.0959804, -0.8171915),
        T(-1.2780054, 0.23534934, -0.5110631, 1.4640164,
          -0.35087395, 0.25458562, -0.65749437, -1.2743273,
          0.32705495, -0.92280126, 1.8714464, -1.4813888,
          0.60687494, 0.33935368, 1.2551553, -0.71658915),
        T(-0.77465796, 0.3714019, -0.45245644, 1.1980948,
          0.32732904, -0.21935141, -0.19621742, -1.0291177,
          0.11420453, -0.9387212, 1.9743775, -2.0302484,
          0.5383579, 1.1113635, 0.4704703, -0.599585),
        T(-0.8154292, 0.6225467, -0.6287141, 1.2946622,
          0.2809173, -0.40715468, -0.28501934, -1.2143513,
          0.49433085, -0.80149204, 2.1966798, -1.8524576,
          0.574358, 1.0066259, 0.9082394, -0.5721369)),
        T(
          T(1.3018701, 1.599837, -0.66893494, -0.4747932,
          1.4816235, -2.1988738, 0.80288893, -0.08690637,
          1.4123986, 0.8986485, 0.9886181, -0.05089509,
          -0.1168761, 1.2418115, 0.11437249, 0.59445935),
          T(1.2257497, 1.644748, -0.7198268, -0.38725024,
            1.4432738, -2.2042546, 0.7490996, -0.17839986,
            1.4885247, 0.8769602, 1.1051279, -0.08014816,
            -0.08212907, 1.2243906, 0.2546038, 0.5629713),
          T(-1.2719716, 0.16566879, -0.47614235, 1.0029007,
            -1.2952622, 0.62499654, -1.0790219, -1.025868,
            0.94064814, -0.07148974, 0.6702734, 0.7035586,
            0.3409673, -1.3214715, 2.244514, -0.3729545),
          T(0.81476617, 1.4433428, -0.71186674, -0.31119436,
            0.63634753, -1.6565179, 0.27403653, -0.26883018,
            1.7255578, 1.0771092, 0.64149255, 0.90875554,
            -0.10430496, 0.16574204, 1.0562141, 0.5474888),
          T(-1.285131, 0.17195511, -0.47159803, 1.4530414,
            -0.35627913, 0.3122788, -0.64908904, -1.2403321,
            0.24032426, -0.96024, 1.8233403, -1.5194026,
            0.6023792, 0.34928182, 1.171351, -0.7291994),
          T(-1.098374, 0.00385495, -0.30677474, 1.2947041,
            -0.10549277, 0.3229493, -0.43214977, -1.0127946,
            -0.14591314, -1.0853384, 1.6767416, -1.848886,
            0.55326104, 0.6819846, 0.56096387, -0.71889424))))

    paramsTable = transformer.getParametersTable()
    for (i <- paramsTable.keySet) {
      if (i.toString contains "embedding") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](T(T( 0.0901417, -0.25920567, 0.35886005, -0.79801846),
            T( 0.7101189, -0.5279109, 0.24793072, 0.07292826),
            T(-0.00906177, 0.6962627, 0.37465635, 0.15718417),
            T(-0.11258064, -0.3311236, -0.3180385, 0.58970255),
            T( 0.17320412, -0.49935055, 0.7124023, -0.28340986),
            T(-0.33200186, 1.0381325, -0.18797834, 0.5976197),
            T(-0.06744625, -0.23964763, 0.37403554, -0.4539435),
            T(-0.39824682, 0.18769431, 0.02896992, -0.7393345),
            T( 0.5590472, -0.7522993, -0.44121778, -0.1815617),
            T( 0.7071572, 0.27919358, 0.23945637, -0.17475012),
            T(-0.36576417, -1.6407981, -0.5480189, 0.00637588),
            T( 0.3870772, 0.5724747, -0.6339975, -0.6118532),
            T(-0.08697614, -0.21675488, -0.13310283, 0.19130157),
            T( 0.06459922, -0.57852674, 0.9070809, 0.09887356),
            T( 0.8016945, -0.09532502, -0.6059104, 0.74728966),
            T( 0.24903144, 0.06780083, 0.16405171, -0.29252014))
          )
        )
      }
    }

    val input1 = Tensor[Float](T(T(3, 1, 2, 3, 4, 5), T(6, 7, 8, 9, 10, 11))).add(1.0f)
    val input2 = Tensor[Float](T(T(4, 5, 7, 9, 10, 11), T(4, 12, 6, 3, 2, 15))).add(1.0f)
    val output = transformer.forward(T(input1, input2)).toTensor[Float]

    require(output.almostEqual(expectedOutput, 1e-5) == true)

    val gradInput = transformer.backward(T(input1, input2), output)
  }

  "AttentionBiasConstant" should "work correctly" in {
    val layer = new PositionEncode[Float]()

    val input = Tensor[Float](T(T(
      T(1.5575712, 1.6023955, 1.4487493, 0.46178865),
      T(1.4542825, 0.36078143, 1.0112681, 1.7850459),
      T(1.0922418, 1.8467345, 0.17114377, 1.5875602),
      T(1.3181713, 1.1110513, 0.31925488, 0.61749554),
      T(0.30953693, 0.93909645, 1.9877799, 1.2225482),
      T(1.3529022, 0.3599646, 1.3499286, 0.4491992)),
      T(T(0.10186243, 0.9201369, 1.6568646, 0.47073865),
        T(1.950448, 1.6722536, 0.5169549, 0.83770823),
        T(1.4055192, 1.535857, 1.0745583, 1.4468269),
        T(0.53809, 0.01234245, 0.06532454, 0.1288917),
        T(1.6856189, 1.4987106, 0.1509037, 1.2490149),
        T(0.6981592, 1.1585901, 1.1459568, 0.3643551))))
    val output = layer.forward(input)

    val outputExpected = Tensor[Float](
      T(T( 0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 1.0000000e+00),
        T( 8.4147096e-01, 9.9999990e-05, 5.4030228e-01, 1.0000000e+00),
        T( 9.0929741e-01, 1.9999998e-04, -4.1614681e-01, 1.0000000e+00),
        T( 1.4112000e-01, 2.9999996e-04, -9.8999250e-01, 9.9999994e-01),
        T(-7.5680250e-01, 3.9999996e-04, -6.5364361e-01, 9.9999994e-01),
        T(-9.5892429e-01, 4.9999997e-04, 2.8366220e-01, 9.9999988e-01))
    )

    output should be(outputExpected)
  }

  "transformer prepare decode layer" should "work correctly" in {
    val prepare = new PositionEncodeWithShift[Float]()

    val input = Tensor[Float](
        T(T(T( 16.24345364, -6.11756414, -5.28171752, -10.72968622),
        T(8.65407629, -23.01538697, 17.44811764, -7.61206901),
        T(3.19039096, -2.49370375, 14.62107937, -20.60140709),
        T(-3.22417204, -3.84054355, 11.33769442, -10.99891267),
        T(-1.72428208, -8.77858418, 0.42213747, 5.82815214),
        T(-11.00619177, 11.4472371, 9.01590721, 5.02494339)),
        T(T(9.00855949, -6.83727859, -1.22890226, -9.35769434),
        T(-2.6788808, 5.30355467, -6.91660752, -3.96753527),
        T(-6.871727, -8.45205641, -6.71246131, -0.12664599),
        T(-11.17310349, 2.34415698, 16.59802177, 7.42044161),
        T(-1.91835552, -8.87628964, -7.47158294, 16.92454601),
        T(0.50807755, -6.36995647, 1.90915485, 21.00255136))))

    val expectedOutput = Tensor[Float](
        T(T(T(0, 0, 1, 1),
        T(17.084925, -6.117464, -4.741415, -9.729686),
        T(9.563374, -23.015186, 17.031971, -6.612069),
        T(3.331511, -2.493404, 13.631087, -19.601408),
        T(-3.9809747, -3.8401434, 10.684051, -9.998913),
        T(-2.6832063, -8.778085, 0.7057997, 6.828152)),
        T(T(0, 0, 1, 1),
        T(9.85003, -6.837178, -0.68859994, -8.357695),
        T(-1.7695832, 5.3037543, -7.332754, -2.9675353),
        T(-6.730607, -8.4517565, -7.702454, 0.87335396),
        T(-11.929906, 2.344557, 15.944379, 8.420442),
        T(-2.8772798, -8.87579, -7.1879206, 17.924545))))

    val expectedGradInput = Tensor[Float](
      T(T(T(17.084925, -6.117464, -4.741415, -9.729686),
        T(9.563374, -23.015186, 17.031971, -6.612069),
        T(3.331511, -2.493404, 13.631087, -19.601408),
        T(-3.9809747, -3.8401434, 10.684051, -9.998913),
        T(-2.6832063, -8.778085, 0.7057997, 6.828152),
        T(0, 0, 0, 0)),
      T(T(9.85003, -6.837178, -0.68859994, -8.357695),
        T(-1.7695832, 5.3037543, -7.332754, -2.9675353),
        T(-6.730607, -8.4517565, -7.702454, 0.87335396),
        T(-11.929906, 2.344557, 15.944379, 8.420442),
        T(-2.8772798, -8.87579, -7.1879206, 17.924545),
        T(0, 0, 0, 0))))

    val out = prepare.forward(input)
    out should be(expectedOutput)

    val out2 = prepare.backward(input, out)
    out2 should be(expectedGradInput)

  }

  "SelfAttentionBiasConstant layer" should "work correctly" in {
    val prepare = new SelfAttentionMask[Float]()
    val input = Tensor[Float](T(T(
        T( 16.24345364, -6.11756414, -5.28171752, -10.72968622),
        T(  8.65407629, -23.01538697, 17.44811764, -7.61206901),
        T(  3.19039096, -2.49370375, 14.62107937, -20.60140709),
        T( -3.22417204, -3.84054355, 11.33769442, -10.99891267),
        T( -1.72428208, -8.77858418, 0.42213747, 5.82815214),
        T(-11.00619177, 11.4472371, 9.01590721, 5.02494339)),
        T(T(  9.00855949, -6.83727859, -1.22890226, -9.35769434),
        T( -2.6788808, 5.30355467, -6.91660752, -3.96753527),
        T( -6.871727, -8.45205641, -6.71246131, -0.12664599),
        T(-11.17310349, 2.34415698, 16.59802177, 7.42044161),
        T( -1.91835552, -8.87628964, -7.47158294, 16.92454601),
        T(  0.50807755, -6.36995647, 1.90915485, 21.00255136))))

    val expectedOutput = Tensor[Float](
      T(T(T(T(0.0f, -1e9f, -1e9f, -1e9f, -1e9f, -1e9f),
      T(0.0f, 0.0f, -1e9f, -1e9f, -1e9f, -1e9f),
      T(0.0f, 0.0f, 0.0f, -1e9f, -1e9f, -1e9f),
      T(0.0f, 0.0f, 0.0f, 0.0f, -1e9f, -1e9f),
      T(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1e9f),
      T(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)))))

    val out = prepare.forward(input)
    out should be(expectedOutput)

    val out2 = prepare.backward(input, out)
  }

  "TransformerOperation getPaddingBias" should "work good" in {
    val input = Tensor[Float](T(0, 1, 2, 3, 4, 5, 6, 7)).resize(Array(2, 4))
    val ops = TransformerOperation.getPaddingBias(input)
    val opsExpected = Tensor[Float](T(-1e9f, 0.0f, 0f, 0f, 0f, 0f, 0f, 0f))
      .resize(Array(2, 1, 1, 4))
    ops should be(opsExpected)
  }

  "Split tensor" should "be ok" in {
    val l1 = Tensor[Float](Array[Float](1, 2, 3, 4, 5, 6,
      1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f), Array(2, 6))
    val l2 = Tensor[Float](Array[Float](1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f,
      1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f), Array(2, 6))
    val input = T(l1, l2)

    val layer = new JoinTable[Float](1, -1)
    val output = layer.forward(input).toTensor[Float]

    val layer2 = new SplitTensor[Float](1, 2)
    val o2 = layer2.forward(output)

    val g1 = o2[Tensor[Float]](1)
    val g2 = o2[Tensor[Float]](2)
    assert(g1.almostEqual(l1, 1e-8) == true)
    assert(g2.almostEqual(l2, 1e-8) == true)

    val gradInput = layer2.backward(output, o2)
    assert(output.almostEqual(gradInput, 1e-8) == true)
  }


}

class SelfAttentionMaskSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val model = new SelfAttentionMask[Float]().setName("SelfAttentionMask")
    val input = Tensor[Float](2, 6, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(model, input)
  }
}

class PaddingMaskSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val model = new PaddingMask[Float]().setName("PaddingMask")
    val input = Tensor[Float](2, 6, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(model, input)
  }
}

class PositionEncodeWithShiftSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val model = new PositionEncodeWithShift[Float]().setName("PositionEncodeWithShift")
    val input = Tensor[Float](2, 6, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(model, input)
  }
}

class PositionEncodeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val model = new PositionEncode[Float]().setName("PositionEncode")
    val input = Tensor[Float](2, 6, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(model, input)
  }
}

class SplitTensorSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val l1 = Tensor[Float](Array[Float](1, 2, 3, 4, 5, 6,
      1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f), Array(2, 6))
    val l2 = Tensor[Float](Array[Float](1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f,
      1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f), Array(2, 6))
    val input = T(l1, l2)

    val layer = new JoinTable[Float](1, -1)
    val output = layer.forward(input).toTensor[Float]

    val model = new SplitTensor[Float](1, 2)
    runSerializationTest(model, output)
  }
}

class TransformerSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val hiddenSize = 4
    val numHeads = 2
    val filterSize = 3
    val num_hidden_layers = 2
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val model = Transformer[Float](20,
      hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout).setName("Transformer")
    val input = Tensor[Float](2, 6).apply1(_ => Random.nextInt(10) + 1)
    runSerializationTest(model, input)
  }
}
