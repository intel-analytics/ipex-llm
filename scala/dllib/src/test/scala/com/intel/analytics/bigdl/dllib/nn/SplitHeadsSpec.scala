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
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class SplitHeadsSpec extends FlatSpec with Matchers {
    val inputX : Tensor[Float] = Tensor(
      T(T(T( 0.81217268, -0.30587821, -0.26408588, -0.53648431, 0.43270381,
      -1.15076935, 0.87240588, -0.38060345),
      T( 0.15951955, -0.12468519, 0.73105397, -1.03007035, -0.1612086,
        -0.19202718, 0.56688472, -0.54994563),
      T(-0.0862141, -0.43892921, 0.02110687, 0.29140761, -0.55030959,
      0.57236185, 0.45079536, 0.25124717)),
      T(T( 0.45042797, -0.34186393, -0.06144511, -0.46788472, -0.13394404,
      0.26517773, -0.34583038, -0.19837676),
      T(-0.34358635, -0.42260282, -0.33562307, -0.0063323, -0.55865517,
      0.11720785, 0.82990109, 0.37102208),
      T(-0.09591778, -0.44381448, -0.37357915, 0.8462273, 0.02540388,
        -0.31849782, 0.09545774, 1.05012757))))

    val outputExpected : Tensor[Float] = Tensor[Float](
      T(T(T(T( 0.8121727, -0.3058782),
        T( 0.15951955, -0.12468519),
        T(-0.0862141, -0.4389292)),
        T(T(-0.2640859, -0.5364843),
          T( 0.73105395, -1.0300703),
          T( 0.02110687, 0.29140761)),
        T(T( 0.43270382, -1.1507694),
          T(-0.1612086, -0.19202718),
          T(-0.5503096, 0.5723618)),
        T(T( 0.8724059, -0.38060346),
          T( 0.5668847, -0.54994565),
          T( 0.45079535, 0.25124717))),
        T(T(T( 0.45042798, -0.34186393),
          T(-0.34358636, -0.42260283),
          T(-0.09591778, -0.4438145)),
          T(T(-0.06144511, -0.46788472),
            T(-0.33562306, -0.0063323),
            T(-0.37357914, 0.8462273)),
          T(T(-0.13394403, 0.26517773),
            T(-0.5586552, 0.11720785),
            T( 0.02540388, -0.31849784)),
          T(T(-0.34583038, -0.19837676),
            T( 0.8299011, 0.37102208),
            T( 0.09545774, 1.0501276))))
    )

    val gradInputExpected : Tensor[Float] = Tensor[Float](
      T(T(T( 0.06007948, 0.30860156, -0.10444712, 0.2933116, -0.37719896,
      0.6264341, 0.5658147, 0.75990844),
      T( 0.15008517, -0.17612493, 0.4194917, 0.46555105, 0.2564649,
        -0.14904642, 1.0927877, -0.69824815),
      T(-0.5712591, -0.17467137, 0.14279366, 0.4425706, 0.24425907,
        -0.03778586, -0.7220569, -0.25223294)),

      T(T( 0.08001854, 0.43808445, 0.11504737, 0.3810056, 0.09914986,
      0.05950432, 0.59945893, 0.09257821),
      T(0.15781747, -1.0111006, -0.11116407, -0.10037903, -0.33533114,
      0.18878189, -0.18764247, -0.3193652),
      T(-0.15310201, 0.4139873, 0.0932807, 0.20502582, 0.06091063,
      0.56474197, 0.21174718, 0.03867003)))
    )

    val gradOutput = Tensor[Float](
      T(T(T(T( 0.06007948, 0.30860155),
        T( 0.15008516, -0.17612492),
        T(-0.5712591, -0.17467136)),

        T(T(-0.10444712, 0.2933116),
          T( 0.41949171, 0.46555104),
          T( 0.14279366, 0.44257058)),

        T(T(-0.37719897, 0.62643408),
          T( 0.25646491, -0.14904642),
          T( 0.24425907, -0.03778586)),

        T(T( 0.56581469, 0.75990841),
          T( 1.0927877, -0.69824817),
          T(-0.7220569, -0.25223293))),

        T(T(T( 0.08001853, 0.43808446),
          T( 0.15781747, -1.01110061),
          T(-0.15310201, 0.41398732)),

          T(T( 0.11504737, 0.38100559),
            T(-0.11116407, -0.10037903),
            T( 0.0932807, 0.20502582)),

          T(T( 0.09914986, 0.05950432),
            T(-0.33533114, 0.18878189),
            T( 0.06091064, 0.56474195)),

          T(T( 0.59945894, 0.09257821),
            T(-0.18764248, -0.3193652),
            T( 0.21174718, 0.03867003))))
    )

  "split heads layer" should "work correctly" in {
    val layer = new SplitHeads[Float](8, 4)

    val output = layer.forward(inputX)
    val gradInput = layer.backward(inputX, gradOutput)

    output should  be(outputExpected)
    gradInput should be(gradInputExpected)
  }

  "split heads layer with mul" should "work correctly" in {
    val layer = new SplitHeads[Float](8, 4, mul = true)

    val output = layer.forward(inputX)
    val gradInput = layer.backward(inputX, gradOutput)

    val outputExpected = Tensor[Float](
      T(T(T(T( 0.57429284, -0.21628854),
        T( 0.11279736, -0.08816575),
        T(-0.06096258, -0.31036982)),
        T(T(-0.18673693, -0.37935168),
          T( 0.5169332, -0.7283697),
          T( 0.01492481, 0.2060563)),
        T(T( 0.3059678, -0.8137168),
          T(-0.11399169, -0.13578372),
          T(-0.38912764, 0.40472093)),
        T(T( 0.6168841, -0.26912728),
          T( 0.400848, -0.3888703),
          T( 0.31876045, 0.17765857))),
        T(T(T( 0.31850067, -0.2417343),
          T(-0.24295224, -0.29882532),
          T(-0.06782411, -0.31382424)),
          T(T(-0.04344825, -0.33084446),
            T(-0.23732133, -0.00447761),
            T(-0.26416034, 0.59837306)),
          T(T(-0.09471273, 0.18750897),
            T(-0.3950289, 0.08287846),
            T( 0.01796325, -0.22521198)),
          T(T(-0.24453901, -0.14027356),
            T( 0.5868287, 0.26235223),
            T( 0.06749881, 0.74255234))))
    )
    val gradInputExpected = Tensor[Float](
      T(T(T( 0.04248261, 0.21821424, -0.07385527, 0.20740262, -0.26671994,
        0.4429558, 0.40009138, 0.5373364),
        T( 0.10612623, -0.12453913, 0.29662544, 0.3291943, 0.18134807,
          -0.10539173, 0.7727176, -0.493736),
        T(-0.40394115, -0.12351131, 0.10097036, 0.31294465, 0.17271724,
          -0.02671864, -0.51057136, -0.17835562)),
        T(T( 0.05658165, 0.3097725, 0.08135077, 0.26941162, 0.07010954,
        0.04207591, 0.42388147, 0.06546268),
        T( 0.1115938, -0.7149561, -0.07860487, -0.07097869, -0.23711492,
        0.13348895, -0.13268326, -0.2258253),
        T(-0.10825947, 0.29273322, 0.06595941, 0.14497514, 0.04307032,
        0.39933288, 0.14972787, 0.02734384)))
    )
    output should  be(outputExpected)
    gradInput should be(gradInputExpected)
  }
}

class SplitHeadsSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = new SplitHeads[Float](8, 4, mul = true).setName("splitheads")
    val input = Tensor[Float](2, 3, 8).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
