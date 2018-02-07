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

import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.math.abs
import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class SpatialMaxPoolingSpec extends FlatSpec with Matchers {

  "A SpatialMaxPooling" should "generate correct output and gradInput" in {
    val module = new SpatialMaxPooling[Double](2, 2)
    val input = Tensor[Double](1, 3, 3)
    input(Array(1, 1, 1)) = 0.53367262030952
    input(Array(1, 1, 2)) = 0.79637692729011
    input(Array(1, 1, 3)) = 0.56747663160786
    input(Array(1, 2, 1)) = 0.18039962812327
    input(Array(1, 2, 2)) = 0.24608615692705
    input(Array(1, 2, 3)) = 0.22956256521866
    input(Array(1, 3, 1)) = 0.30736334621906
    input(Array(1, 3, 2)) = 0.59734606579877
    input(Array(1, 3, 3)) = 0.42989541869611
    val gradOutput = Tensor[Double](1, 1, 1)
    gradOutput(Array(1, 1, 1)) = 0.023921491578221
    val expectedOutput = Tensor[Double](1, 1, 1)
    expectedOutput(Array(1, 1, 1)) = 0.79637692729011
    val expectedGrad = Tensor[Double](1, 3, 3)
    expectedGrad(Array(1, 1, 1)) = 0
    expectedGrad(Array(1, 1, 2)) = 0.023921491578221
    expectedGrad(Array(1, 1, 3)) = 0
    expectedGrad(Array(1, 2, 1)) = 0
    expectedGrad(Array(1, 2, 2)) = 0
    expectedGrad(Array(1, 2, 3)) = 0
    expectedGrad(Array(1, 3, 1)) = 0
    expectedGrad(Array(1, 3, 2)) = 0
    expectedGrad(Array(1, 3, 3)) = 0
    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    expectedOutput.map(output, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)
  }

  "A SpatialMaxPooling of float" should "generate correct output and gradInput" in {
    val module = new SpatialMaxPooling[Float](2, 2)
    val input = Tensor[Float](1, 3, 3)
    input(Array(1, 1, 1)) = 0.53367262030952f
    input(Array(1, 1, 2)) = 0.79637692729011f
    input(Array(1, 1, 3)) = 0.56747663160786f
    input(Array(1, 2, 1)) = 0.18039962812327f
    input(Array(1, 2, 2)) = 0.24608615692705f
    input(Array(1, 2, 3)) = 0.22956256521866f
    input(Array(1, 3, 1)) = 0.30736334621906f
    input(Array(1, 3, 2)) = 0.59734606579877f
    input(Array(1, 3, 3)) = 0.42989541869611f
    val gradOutput = Tensor[Float](1, 1, 1)
    gradOutput(Array(1, 1, 1)) = 0.023921491578221f
    val expectedOutput = Tensor[Float](1, 1, 1)
    expectedOutput(Array(1, 1, 1)) = 0.79637692729011f
    val expectedGrad = Tensor[Float](1, 3, 3)
    expectedGrad(Array(1, 1, 1)) = 0
    expectedGrad(Array(1, 1, 2)) = 0.023921491578221f
    expectedGrad(Array(1, 1, 3)) = 0
    expectedGrad(Array(1, 2, 1)) = 0
    expectedGrad(Array(1, 2, 2)) = 0
    expectedGrad(Array(1, 2, 3)) = 0
    expectedGrad(Array(1, 3, 1)) = 0
    expectedGrad(Array(1, 3, 2)) = 0
    expectedGrad(Array(1, 3, 3)) = 0
    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    expectedOutput.map(output, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)
  }

  "A SpatialMaxPooling in ceil mode" should "generate correct output and gradInput" in {
    val module = new SpatialMaxPooling[Double](2, 2).ceil()
    val input = Tensor[Double](1, 3, 3)
    input(Array(1, 1, 1)) = 0.52796983974986
    input(Array(1, 1, 2)) = 0.77937559061684
    input(Array(1, 1, 3)) = 0.35681968415156
    input(Array(1, 2, 1)) = 0.11191992252134
    input(Array(1, 2, 2)) = 0.18749328423291
    input(Array(1, 2, 3)) = 0.56990833440796
    input(Array(1, 3, 1)) = 0.40316500258632
    input(Array(1, 3, 2)) = 0.081480369903147
    input(Array(1, 3, 3)) = 0.9328313653823
    val gradOutput = Tensor[Double](1, 2, 2)
    gradOutput(Array(1, 1, 1)) = 0.032919396646321
    gradOutput(Array(1, 1, 2)) = 0.54643597058021
    gradOutput(Array(1, 2, 1)) = 0.15019989991561
    gradOutput(Array(1, 2, 2)) = 0.29359312891029
    val expectedOutput = Tensor[Double](1, 2, 2)
    expectedOutput(Array(1, 1, 1)) = 0.77937559061684
    expectedOutput(Array(1, 1, 2)) = 0.56990833440796
    expectedOutput(Array(1, 2, 1)) = 0.40316500258632
    expectedOutput(Array(1, 2, 2)) = 0.9328313653823
    val expectedGrad = Tensor[Double](1, 3, 3)
    expectedGrad(Array(1, 1, 1)) = 0
    expectedGrad(Array(1, 1, 2)) = 0.032919396646321
    expectedGrad(Array(1, 1, 3)) = 0
    expectedGrad(Array(1, 2, 1)) = 0
    expectedGrad(Array(1, 2, 2)) = 0
    expectedGrad(Array(1, 2, 3)) = 0.54643597058021
    expectedGrad(Array(1, 3, 1)) = 0.15019989991561
    expectedGrad(Array(1, 3, 2)) = 0
    expectedGrad(Array(1, 3, 3)) = 0.29359312891029
    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    expectedOutput.map(output, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)
  }

  "A SpatialMaxPooling with dW,dH and pad" should "generate correct output and gradInput" in {
    val module = new SpatialMaxPooling[Double](3, 2, 2, 2, 1, 1)
    val input = Tensor[Double](1, 4, 4)
    input(Array(1, 1, 1)) = 0.39016278996132
    input(Array(1, 1, 2)) = 0.2185998596251
    input(Array(1, 1, 3)) = 0.25424918020144
    input(Array(1, 1, 4)) = 0.27032148116268
    input(Array(1, 2, 1)) = 0.060930834617466
    input(Array(1, 2, 2)) = 0.45138090639375
    input(Array(1, 2, 3)) = 0.15180362504907
    input(Array(1, 2, 4)) = 0.94037068192847
    input(Array(1, 3, 1)) = 0.59107813285664
    input(Array(1, 3, 2)) = 0.35054552881047
    input(Array(1, 3, 3)) = 0.13943290174939
    input(Array(1, 3, 4)) = 0.55932607129216
    input(Array(1, 4, 1)) = 0.36138460785151
    input(Array(1, 4, 2)) = 0.09617441915907
    input(Array(1, 4, 3)) = 0.0016057807952166
    input(Array(1, 4, 4)) = 0.63314784877002
    val gradOutput = Tensor[Double](1, 3, 2)
    gradOutput(Array(1, 1, 1)) = 0.090864511439577
    gradOutput(Array(1, 1, 2)) = 0.023676526965573
    gradOutput(Array(1, 2, 1)) = 0.14472470176406
    gradOutput(Array(1, 2, 2)) = 0.55145429610275
    gradOutput(Array(1, 3, 1)) = 0.23100447538309
    gradOutput(Array(1, 3, 2)) = 0.7202813832555
    val expectedOutput = Tensor[Double](1, 3, 2)
    expectedOutput(Array(1, 1, 1)) = 0.39016278996132
    expectedOutput(Array(1, 1, 2)) = 0.27032148116268
    expectedOutput(Array(1, 2, 1)) = 0.59107813285664
    expectedOutput(Array(1, 2, 2)) = 0.94037068192847
    expectedOutput(Array(1, 3, 1)) = 0.36138460785151
    expectedOutput(Array(1, 3, 2)) = 0.63314784877002
    val expectedGrad = Tensor[Double](1, 4, 4)
    expectedGrad(Array(1, 1, 1)) = 0.090864511439577
    expectedGrad(Array(1, 1, 2)) = 0
    expectedGrad(Array(1, 1, 3)) = 0
    expectedGrad(Array(1, 1, 4)) = 0.023676526965573
    expectedGrad(Array(1, 2, 1)) = 0
    expectedGrad(Array(1, 2, 2)) = 0
    expectedGrad(Array(1, 2, 3)) = 0
    expectedGrad(Array(1, 2, 4)) = 0.55145429610275
    expectedGrad(Array(1, 3, 1)) = 0.14472470176406
    expectedGrad(Array(1, 3, 2)) = 0
    expectedGrad(Array(1, 3, 3)) = 0
    expectedGrad(Array(1, 3, 4)) = 0
    expectedGrad(Array(1, 4, 1)) = 0.23100447538309
    expectedGrad(Array(1, 4, 2)) = 0
    expectedGrad(Array(1, 4, 3)) = 0
    expectedGrad(Array(1, 4, 4)) = 0.7202813832555
    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    expectedOutput.map(output, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)
  }

  "A SpatialMaxPooling with asymmetry dW,dH and pad" should
    "generate correct output and gradInput" in {
    val module = new SpatialMaxPooling[Double](3, 2, 2, 1, 0, 1)
    val input = Tensor[Double](1, 4, 3)
    input(Array(1, 1, 1)) = 0.33230334869586
    input(Array(1, 1, 2)) = 0.96416000230238
    input(Array(1, 1, 3)) = 0.36518415389583
    input(Array(1, 2, 1)) = 0.20545255555771
    input(Array(1, 2, 2)) = 0.87067628931254
    input(Array(1, 2, 3)) = 0.010092333424836
    input(Array(1, 3, 1)) = 0.24038829724304
    input(Array(1, 3, 2)) = 5.5465148761868e-05
    input(Array(1, 3, 3)) = 0.52843235991895
    input(Array(1, 4, 1)) = 0.85843464848585
    input(Array(1, 4, 2)) = 0.8718444830738
    input(Array(1, 4, 3)) = 0.55945817148313
    val gradOutput = Tensor[Double](1, 5, 1)
    gradOutput(Array(1, 1, 1)) = 0.35993986087851
    gradOutput(Array(1, 2, 1)) = 0.49970405758359
    gradOutput(Array(1, 3, 1)) = 0.64157117158175
    gradOutput(Array(1, 4, 1)) = 0.090919603593647
    gradOutput(Array(1, 5, 1)) = 0.74979126174003
    val expectedOutput = Tensor[Double](1, 5, 1)
    expectedOutput(Array(1, 1, 1)) = 0.96416000230238
    expectedOutput(Array(1, 2, 1)) = 0.96416000230238
    expectedOutput(Array(1, 3, 1)) = 0.87067628931254
    expectedOutput(Array(1, 4, 1)) = 0.8718444830738
    expectedOutput(Array(1, 5, 1)) = 0.8718444830738
    val expectedGrad = Tensor[Double](1, 4, 3)
    expectedGrad(Array(1, 1, 1)) = 0
    expectedGrad(Array(1, 1, 2)) = 0.8596439184621
    expectedGrad(Array(1, 1, 3)) = 0
    expectedGrad(Array(1, 2, 1)) = 0
    expectedGrad(Array(1, 2, 2)) = 0.64157117158175
    expectedGrad(Array(1, 2, 3)) = 0
    expectedGrad(Array(1, 3, 1)) = 0
    expectedGrad(Array(1, 3, 2)) = 0
    expectedGrad(Array(1, 3, 3)) = 0
    expectedGrad(Array(1, 4, 1)) = 0
    expectedGrad(Array(1, 4, 2)) = 0.84071086533368
    expectedGrad(Array(1, 4, 3)) = 0
    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    expectedOutput.map(output, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)
  }

  "A SpatialMaxPooling" should "generate correct output and gradInput with 3D input" in {
    val module = new SpatialMaxPooling[Double](3, 2, 2, 1, 1, 0)
    val input = Tensor[Double](2, 3, 3)
    input(Array(1, 1, 1)) = 0.71901097916998
    input(Array(1, 1, 2)) = 0.77136012469418
    input(Array(1, 1, 3)) = 0.55675543914549
    input(Array(1, 2, 1)) = 0.037243376020342
    input(Array(1, 2, 2)) = 0.56937403371558
    input(Array(1, 2, 3)) = 0.97276814724319
    input(Array(1, 3, 1)) = 0.72952383407392
    input(Array(1, 3, 2)) = 0.75221347878687
    input(Array(1, 3, 3)) = 0.067107456270605
    input(Array(2, 1, 1)) = 0.99181169876829
    input(Array(2, 1, 2)) = 0.68115968839265
    input(Array(2, 1, 3)) = 0.46248584869318
    input(Array(2, 2, 1)) = 0.20620839577168
    input(Array(2, 2, 2)) = 0.64804595755413
    input(Array(2, 2, 3)) = 0.34937693946995
    input(Array(2, 3, 1)) = 0.89873241703026
    input(Array(2, 3, 2)) = 0.36333921016194
    input(Array(2, 3, 3)) = 0.87027320521884
    val gradOutput = Tensor[Double](2, 2, 2)
    gradOutput(Array(1, 1, 1)) = 0.90343235363252
    gradOutput(Array(1, 1, 2)) = 0.15445829788223
    gradOutput(Array(1, 2, 1)) = 0.54508250067011
    gradOutput(Array(1, 2, 2)) = 0.68350423313677
    gradOutput(Array(2, 1, 1)) = 0.77796681411564
    gradOutput(Array(2, 1, 2)) = 0.73553358553909
    gradOutput(Array(2, 2, 1)) = 0.81981368502602
    gradOutput(Array(2, 2, 2)) = 0.67357461876236
    val expectedOutput = Tensor[Double](2, 2, 2)
    expectedOutput(Array(1, 1, 1)) = 0.77136012469418
    expectedOutput(Array(1, 1, 2)) = 0.97276814724319
    expectedOutput(Array(1, 2, 1)) = 0.75221347878687
    expectedOutput(Array(1, 2, 2)) = 0.97276814724319
    expectedOutput(Array(2, 1, 1)) = 0.99181169876829
    expectedOutput(Array(2, 1, 2)) = 0.68115968839265
    expectedOutput(Array(2, 2, 1)) = 0.89873241703026
    expectedOutput(Array(2, 2, 2)) = 0.87027320521884
    val expectedGrad = Tensor[Double](2, 3, 3)
    expectedGrad(Array(1, 1, 1)) = 0
    expectedGrad(Array(1, 1, 2)) = 0.90343235363252
    expectedGrad(Array(1, 1, 3)) = 0
    expectedGrad(Array(1, 2, 1)) = 0
    expectedGrad(Array(1, 2, 2)) = 0
    expectedGrad(Array(1, 2, 3)) = 0.837962531019
    expectedGrad(Array(1, 3, 1)) = 0
    expectedGrad(Array(1, 3, 2)) = 0.54508250067011
    expectedGrad(Array(1, 3, 3)) = 0
    expectedGrad(Array(2, 1, 1)) = 0.77796681411564
    expectedGrad(Array(2, 1, 2)) = 0.73553358553909
    expectedGrad(Array(2, 1, 3)) = 0
    expectedGrad(Array(2, 2, 1)) = 0
    expectedGrad(Array(2, 2, 2)) = 0
    expectedGrad(Array(2, 2, 3)) = 0
    expectedGrad(Array(2, 3, 1)) = 0.81981368502602
    expectedGrad(Array(2, 3, 2)) = 0
    expectedGrad(Array(2, 3, 3)) = 0.67357461876236
    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    expectedOutput.map(output, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)
  }

  "A SpatialMaxPooling of float" should "be good in gradient checker" in {
    RandomGenerator.RNG.setSeed(1000)
    val module = new SpatialMaxPooling[Float](2, 2)
    val input = Tensor[Float](1, 3, 3).rand()
    val checker = new GradientChecker(1e-4, 1e-2)
    checker.checkLayer[Float](module, input) should be(true)
  }

  "A SpatialMaxPooling" should "work with SAME padding using NCHW format" in {
    import tensor.TensorNumericMath.TensorNumeric.NumericFloat

    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = -1
    val padH = -1
    val layer = new SpatialMaxPooling(kW, kH, dW, dH, padW, padH)

    val inputData = Array(
      1.0f, 2, 3, 4
    )

    val input = Tensor(Storage(inputData), 1, Array(1, 2, 2))
    val output = layer.updateOutput(input)
    val gradInput = layer.backward(input, output)
    output.storage().array() should be (Array(4.0f, 4, 4, 4))
    gradInput.storage().array() should be (Array(0.0f, 0, 0, 16))
  }

  "A SpatialMaxPoolingNHWC" should "generate correct output and gradInput" in {

    import tensor.TensorNumericMath.TensorNumeric.NumericDouble
    case class Pooling(kW: Int, kH: Int, dW: Int, dH: Int, pW: Int, pH: Int)
    val params = List(
      Pooling(3, 3, 1, 1, 0, 0),
      Pooling(1, 1, 1, 1, 0, 0),
      Pooling(5, 5, 1, 1, 0, 0),
      Pooling(3, 3, 2, 2, 0, 0),
      Pooling(1, 1, 2, 2, 0, 0),
      Pooling(5, 5, 2, 2, 0, 0),
      Pooling(3, 3, 2, 2, 1, 1),
      Pooling(5, 5, 2, 2, 1, 1),
      Pooling(1, 1, 2, 2, -1, -1),
      Pooling(5, 5, 2, 2, -1, -1),
      Pooling(2, 2, 1, 1, -1, -1)
    )
    for (param <- params) {
      println(param)

      val module = new SpatialMaxPooling(param.kW, param.kH, param.dW, param.dH, param.pW, param.pH)
      val moduleNHWC = new SpatialMaxPooling(param.kW, param.kH, param.dW, param.dH,
        param.pW, param.pH, format = DataFormat.NHWC)

      val input = Tensor(2, 4, 5, 5).randn()

      val inputNHWC = Tensor(input.size()).copy(input)
        .transpose(2, 4).transpose(2, 3).contiguous()

      val expectedOutput = module.forward(input)
      val expectedGrad = module.backward(input, expectedOutput)

      var output = moduleNHWC.forward(inputNHWC)
      var gradInput = moduleNHWC.backward(inputNHWC, output)
      output = output.transpose(2, 4).transpose(3, 4)
      gradInput = gradInput.transpose(2, 4).transpose(3, 4)
      expectedOutput.map(output, (v1, v2) => {
        assert(abs(v1 - v2) < 1e-6)
        v1
      })

      expectedGrad.map(gradInput, (v1, v2) => {
        assert(abs(v1 - v2) < 1e-6)
        v1
      })

    }
  }

}

class SpatialMaxPoolingSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val spatialMaxPooling = SpatialMaxPooling[Float](2, 2, 2, 2).
      setName("spatialMaxPooling")
    val input = Tensor[Float](1, 3, 3).apply1( e => Random.nextFloat())
    runSerializationTest(spatialMaxPooling, input)
  }
}
