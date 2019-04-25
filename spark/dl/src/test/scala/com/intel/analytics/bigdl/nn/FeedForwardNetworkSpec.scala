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

class FeedForwardNetworkSpec extends FlatSpec with Matchers {
  val input : Tensor[Float] = Tensor(T(T(
    T(1.62434536, -0.61175641, -0.52817175, -1.07296862, 0.86540763, -2.3015387,
      1.74481176, -0.7612069),
    T(0.3190391, -0.24937038, 1.46210794, -2.06014071, -0.3224172, -0.38405435,
      1.13376944, -1.09989127),
    T(-0.17242821, -0.87785842, 0.04221375, 0.58281521, -1.10061918, 1.14472371,
      0.90159072, 0.50249434)),

    T(T(0.90085595, -0.68372786, -0.12289023, -0.93576943, -0.26788808, 0.53035547,
      -0.69166075, -0.39675353),
      T(-0.6871727, -0.84520564, -0.67124613, -0.0126646, -1.11731035, 0.2344157,
        1.65980218, 0.74204416),
      T(-0.19183555, -0.88762896, -0.74715829, 1.6924546, 0.05080775, -0.63699565,
        0.19091548, 2.10025514))
  ))

  val outputExpected : Tensor[Float] = Tensor[Float](
    T(T(T(-1.8375108, 1.2966242, -0.7180478, -0.23646069, -0.26609686, 1.5588356,
      0.660595, -0.31564748),
      T(-0.19088337, 0.3883139, -0.07755771, 0.4170658, 0.26731488, -0.17202748,
        -0.09847746, -0.48390502),
      T(-0.83966494, 0.9178331, -1.5419102, -1.1216922, -0.71579254, -0.8785725,
        1.228107, -0.8411994)),
      T(T(-0.05918283, 0.11268676, -0.14456667, -0.07148183, -0.08459917, -0.12416959,
        0.14727916, -0.04476965),
        T(-0.8799405, 0.8714211, -0.9089507, -0.48824388, -0.44684958, -0.03341737,
          0.83358747, -0.3849794),
        T(-1.6615174, 1.5758176, -1.3603796, -0.5994581, -0.6806528, 0.39317507,
          1.3247881, -0.49895704)))
  )

  val gradInputExpected : Tensor[Float] = Tensor[Float](
    T(T(T( 1.5185112, -1.227324, -0.5093635, 1.1931186, 0.3033546, -1.7162423,
      1.6165192, 0.7531797),
      T(-0.34178314, 0.43916014, 0.2897479, -0.10748425, -0.23153473, -0.02656112,
        0.00322444, -0.1791711),
      T( 0.224675, -2.402936, -0.20401537, 2.6879046, -0.9117198, 2.6889753,
        0.36316222, 1.2847843)),
    T(T(-0.04712493, -0.18667392, -0.04214612, 0.22826877, -0.10985573, 0.33849096,
      -0.04582711, 0.07289039),
      T( 0.47275004, -1.6686311, -0.37326545, 1.8608568, -0.47097835, 1.1310074,
        0.54947317, 0.8439914),
      T( 1.0096599, -2.7125025, -0.75489235, 3.013646, -0.6398741, 1.2858341,
        1.1247286, 1.3493999)))
  )

  val gradWeights: Table = T(
      "filter_layer" -> Tensor[Float](T(
        T( 3.0687852, -1.2373898, 0.26932785, -2.203119),
        T(-6.725154, -4.244735, -0.21051459, -7.5355434),
        T(-4.565733, -1.7183428, 1.2342887, -3.5032144),
        T( 2.267335, 3.520794, -1.7391386, 6.6742306),
        T(-0.43024874, -3.5453463, -0.27217957, -5.8415084),
        T(-6.704142, 1.8187529, -0.3242127, 2.5449739),
        T( 9.400441, 3.943234, 0.9571104, 6.649495),
        T( 5.318757, 4.9089594, -0.9285111, 9.345928))
      ),
    "output_layer" -> Tensor[Float](
        T(T(-9.917135, 8.18558, -6.3063927, -2.7818725,
          -2.8578176, 5.0143423, 5.885993, -2.608803),
        T(-2.1594772, 2.2862027, -3.000262, -1.8476318,
          -1.3436246, -1.1266284, 2.4819474, -1.6156014),
        T(-0.15865958, 0.32276106, -0.06446488, 0.34665924,
          0.22218838, -0.14298683, -0.08185308, -0.40221506),
        T(-4.2312956, 4.2519608, -5.0156603, -2.9951196,
          -2.4163678, -0.96183157, 4.4039025, -2.3232994))
    )
  )

  val weights: Table = T(
    "filter_layer" -> Tensor[Float](
      T(T( 0.5093561, -0.07532924, -0.40125486, -0.09511733),
        T(-0.4116828, -0.20966673, 0.53027445, -0.41794384),
        T(-0.17085642, 0.70417756, 0.3094539, -0.44296354),
        T( 0.40020925, 0.07376623, -0.13086122, 0.59578115),
        T( 0.10175461, -0.07514799, -0.27066603, -0.26833212),
        T(-0.57568127, 0.6374385, -0.06203693, 0.6385146),
        T( 0.542231, 0.06174886, 0.00085795, -0.15512145),
        T( 0.25264, 0.4526841, -0.23395362, -0.00881493))
    ),
    "output_layer" -> Tensor[Float](
      T(T(-0.63213325, 0.44605953, -0.24701998, -0.08134627, -0.09154159, 0.5362645,
      0.22725528, -0.1085878),
      T(-0.198219, -0.06108004, -0.41906577, -0.5969921, -0.06956118, -0.16365921,
      0.07787138, -0.49795625),
      T(-0.16827011, 0.4860949, 0.03646076, 0.6866401, 0.34314734, -0.1562866,
        -0.14259237, -0.42798606),
      T(-0.2204194, 0.5318032, -0.5550728, -0.17479968, -0.36973962, -0.52694166,
      0.6547342, -0.0777601)))
  )

  "FeedForwardNetwork layer" should "work correctly" in {
    // compare with tensorflow 1.13.1
    val ffn = new FeedForwardNetwork[Float](8, 4, 1.0f)

    val paramsTable = ffn.getParametersTable()
    val w1 = weights.get[Tensor[Float]]("filter_layer").get
    val w2 = weights.get[Tensor[Float]]("output_layer").get
    for (i <- paramsTable.keySet) {
      val params = paramsTable.get[Table](i).get.get[Tensor[Float]]("weight").get
      if (params.size(1) == w1.size(2)) {
        params.copy(w1.transpose(1, 2))
      } else if (params.size(1) == w2.size(2)) {
        params.copy(w2.transpose(1, 2))
      }
    }

    val output = ffn.forward(input)
    val gradInput = ffn.backward(input, output)

    output should  be(outputExpected)
    gradInput should be(gradInputExpected)

    val gw1 = gradWeights.get[Tensor[Float]]("filter_layer").get
    val gw2 = gradWeights.get[Tensor[Float]]("output_layer").get
    for (i <- paramsTable.keySet) {
      val params = paramsTable.get[Table](i).get.get[Tensor[Float]]("gradWeight").get
      if (params.size(1) == gw1.size(2)) {
        params should be(gw1.transpose(1, 2))
      } else if (params.size(1) == gw2.size(2)) {
        params should be(gw2.transpose(1, 2))
      }
    }
  }
}

class FeedForwardNetworkSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val ffn = new FeedForwardNetwork[Float](8, 4, 1.0f)
    val input = Tensor[Float](2, 3, 8).apply1(_ => Random.nextFloat())
    runSerializationTest(ffn, input)
  }
}
