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
  "tranformer block" should "work correctly" in {
    val vocabSize = 10
    val hiddenSize = 4
    val numHeads = 2
    val filterSize = 3
    val num_hidden_layers = 1
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val transformer = new TransformerLayer[Float](vocabSize,
      hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    val block = transformer.decode(num_hidden_layers)
    val paramsTable = block.getParametersTable()

    for (i <- paramsTable.keySet) {
      if (i == "q") {
        paramsTable.get[Table]("q").get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.12254566, -0.3492695, 0.6760147, 0.4690166),
            T(-0.70616156, -0.7172935, -0.70902413, -0.7268282),
            T(-0.17867321, 0.03752673, 0.21406537, -0.84105927),
            T(-0.40054652, 0.01422167, 0.49654406, -0.62966037))).t())
      } else if (i == "k") {
        paramsTable.get[Table]("k").get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.80201703, 0.29880065, 0.8191585, 0.393151),
            T(-0.43785518, 0.02502167, -0.85530514, 0.86387163),
            T( 0.07737422, 0.34640843, 0.5547114, 0.12658376),
            T( 0.6287202, -0.7140273, -0.08061278, -0.3983137))).t())
      } else if (i == "v") {
        paramsTable.get[Table]("v").get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.14568096, 0.8488055, -0.38585222, -0.42583144),
            T(-0.35776895, 0.00440949, 0.76952034, 0.7039148),
            T(-0.4635923, -0.5273898, 0.36311775, 0.21081167),
            T(-0.04171634, 0.24859089, 0.03242427, -0.01675642))).t())
      } else if (i == "output_transform") {
        paramsTable.get[Table]("output_transform").get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T( 0.8254406, 0.7399195, -0.76593506, -0.38950253),
              T( 0.51289314, 0.1285783, -0.24543494, -0.7138509),
              T(-0.34158242, -0.37842813, -0.5111934, 0.5966528),
              T( 0.39076942, -0.7022542, 0.8254971, -0.50844))).t())
      } else if (i == "filter_layer") {
        paramsTable.get[Table]("filter_layer").get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T( 0.4929167, -0.5465611, 0.4262464),
              T( 0.5161569, -0.6786176, 0.37465477),
              T( 0.35582626, 0.43647707, -0.23218763),
              T( 0.7624726, 0.28653884, 0.20991063))).transpose(1, 2))
      } else if (i == "output_layer") {
        paramsTable.get[Table]("output_layer").get[Tensor[Float]]("weight").copy(
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

    val gradInput = block.backward(T(input, bias), output)

    println("done")
  }

  "transformer prepare decode layer" should "work correctly" in {
    val prepare = new TransformerPrepareDecoder[Float]()

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

  "transformer constant layer" should "work correctly" in {
    val prepare = new TransformerConstant[Float]()
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

}

class TransformerConstantSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val model = new TransformerConstant[Float]().setName("TransformerConstant")
    val input = Tensor[Float](2, 6, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(model, input)
  }
}

class TransformerPrepareDecoderSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val model = new TransformerPrepareDecoder[Float]().setName("TransformerPrepareDecoder")
    val input = Tensor[Float](2, 6, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(model, input)
  }
}

class TransformerLayerSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val hiddenSize = 4
    val numHeads = 2
    val filterSize = 3
    val num_hidden_layers = 2
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val model = new TransformerLayer[Float](20,
      hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)
    val input = Tensor[Float](2, 6).apply1(_ => Random.nextInt(10) + 1)
    runSerializationTest(model, input)
  }
}
