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

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.math.abs
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class SpatialAveragePoolingSpec extends FlatSpec with Matchers {
  "A SpatialAveragePooling" should "generate correct output and gradInput" in {
    val module = new SpatialAveragePooling[Double](3, 2, 2, 1)
    val input = Tensor[Double](1, 4, 3)
    input(Array(1, 1, 1)) = 0.25434372201562
    input(Array(1, 1, 2)) = 0.20443214406259
    input(Array(1, 1, 3)) = 0.33442943682894
    input(Array(1, 2, 1)) = 0.051310112234205
    input(Array(1, 2, 2)) = 0.56103343307041
    input(Array(1, 2, 3)) = 0.041837680386379
    input(Array(1, 3, 1)) = 0.75616162386723
    input(Array(1, 3, 2)) = 0.35945181339048
    input(Array(1, 3, 3)) = 0.4502888196148
    input(Array(1, 4, 1)) = 0.14862711215392
    input(Array(1, 4, 2)) = 0.050680571002886
    input(Array(1, 4, 3)) = 0.93014938035049
    val gradOutput = Tensor[Double](1, 3, 1)
    gradOutput(Array(1, 1, 1)) = 0.22147525195032
    gradOutput(Array(1, 2, 1)) = 0.30394183006138
    gradOutput(Array(1, 3, 1)) = 0.77438542619348
    val expectedOutput = Tensor[Double](1, 3, 1)
    expectedOutput(Array(1, 1, 1)) = 0.24123108809969
    expectedOutput(Array(1, 2, 1)) = 0.37001391376058
    expectedOutput(Array(1, 3, 1)) = 0.44922655339663
    val expectedGrad = Tensor[Double](1, 4, 3)
    expectedGrad(Array(1, 1, 1)) = 0.036912541991721
    expectedGrad(Array(1, 1, 2)) = 0.036912541991721
    expectedGrad(Array(1, 1, 3)) = 0.036912541991721
    expectedGrad(Array(1, 2, 1)) = 0.087569513668617
    expectedGrad(Array(1, 2, 2)) = 0.087569513668617
    expectedGrad(Array(1, 2, 3)) = 0.087569513668617
    expectedGrad(Array(1, 3, 1)) = 0.17972120937581
    expectedGrad(Array(1, 3, 2)) = 0.17972120937581
    expectedGrad(Array(1, 3, 3)) = 0.17972120937581
    expectedGrad(Array(1, 4, 1)) = 0.12906423769891
    expectedGrad(Array(1, 4, 2)) = 0.12906423769891
    expectedGrad(Array(1, 4, 3)) = 0.12906423769891
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

  "A SpatialAveragePooling" should "generate correct output and gradInput with 4D input" in {
    val module = new SpatialAveragePooling[Double](2, 2)
    val input = Tensor[Double](2, 1, 3, 3)
    input(Array(1, 1, 1, 1)) = 0.026265420718119
    input(Array(1, 1, 1, 2)) = 0.99608502909541
    input(Array(1, 1, 1, 3)) = 0.38350357743911
    input(Array(1, 1, 2, 1)) = 0.81032626936212
    input(Array(1, 1, 2, 2)) = 0.33959551295266
    input(Array(1, 1, 2, 3)) = 0.18254237202927
    input(Array(1, 1, 3, 1)) = 0.20723931328394
    input(Array(1, 1, 3, 2)) = 0.45142468763515
    input(Array(1, 1, 3, 3)) = 0.37776186596602
    input(Array(2, 1, 1, 1)) = 0.45621562120505
    input(Array(2, 1, 1, 2)) = 0.30205155536532
    input(Array(2, 1, 1, 3)) = 0.50242692162283
    input(Array(2, 1, 2, 1)) = 0.83084010914899
    input(Array(2, 1, 2, 2)) = 0.047547404188663
    input(Array(2, 1, 2, 3)) = 0.22663795156404
    input(Array(2, 1, 3, 1)) = 0.27236858918332
    input(Array(2, 1, 3, 2)) = 0.9949026464019
    input(Array(2, 1, 3, 3)) = 0.0028261682018638
    val gradOutput = Tensor[Double](2, 1, 2, 2)
    gradOutput(Array(1, 1, 1, 1)) = 0.40912644844502
    gradOutput(Array(1, 1, 1, 2)) = 0.31045490363613
    gradOutput(Array(1, 1, 2, 1)) = 0.81302798143588
    gradOutput(Array(1, 1, 2, 2)) = 0.87783142598346
    gradOutput(Array(2, 1, 1, 1)) = 0.25901150656864
    gradOutput(Array(2, 1, 1, 2)) = 0.25494889705442
    gradOutput(Array(2, 1, 2, 1)) = 0.88176139933057
    gradOutput(Array(2, 1, 2, 2)) = 0.95890677929856
    val expectedOutput = Tensor[Double](2, 1, 2, 2)
    expectedOutput(Array(1, 1, 1, 1)) = 0.54306805803208
    expectedOutput(Array(1, 1, 1, 2)) = 0.47543162287911
    expectedOutput(Array(1, 1, 2, 1)) = 0.45214644580847
    expectedOutput(Array(1, 1, 2, 2)) = 0.33783110964578
    expectedOutput(Array(2, 1, 1, 1)) = 0.40916367247701
    expectedOutput(Array(2, 1, 1, 2)) = 0.26966595818521
    expectedOutput(Array(2, 1, 2, 1)) = 0.53641468723072
    expectedOutput(Array(2, 1, 2, 2)) = 0.31797854258912
    val expectedGrad = Tensor[Double](2, 1, 3, 3)
    expectedGrad(Array(1, 1, 1, 1)) = 0.10228161211126
    expectedGrad(Array(1, 1, 1, 2)) = 0.17989533802029
    expectedGrad(Array(1, 1, 1, 3)) = 0.077613725909032
    expectedGrad(Array(1, 1, 2, 1)) = 0.30553860747023
    expectedGrad(Array(1, 1, 2, 2)) = 0.60261018987512
    expectedGrad(Array(1, 1, 2, 3)) = 0.2970715824049
    expectedGrad(Array(1, 1, 3, 1)) = 0.20325699535897
    expectedGrad(Array(1, 1, 3, 2)) = 0.42271485185483
    expectedGrad(Array(1, 1, 3, 3)) = 0.21945785649586
    expectedGrad(Array(2, 1, 1, 1)) = 0.06475287664216
    expectedGrad(Array(2, 1, 1, 2)) = 0.12849010090576
    expectedGrad(Array(2, 1, 1, 3)) = 0.063737224263605
    expectedGrad(Array(2, 1, 2, 1)) = 0.2851932264748
    expectedGrad(Array(2, 1, 2, 2)) = 0.58865714556305
    expectedGrad(Array(2, 1, 2, 3)) = 0.30346391908824
    expectedGrad(Array(2, 1, 3, 1)) = 0.22044034983264
    expectedGrad(Array(2, 1, 3, 2)) = 0.46016704465728
    expectedGrad(Array(2, 1, 3, 3)) = 0.23972669482464
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

  "A SpatialAveragePooling" should "generate correct output and gradInput with 3D input" in {
    val module = new SpatialAveragePooling[Double](2, 3)
    val input = Tensor[Double](2, 3, 3)
    input(Array(1, 1, 1)) = 0.59898642194457
    input(Array(1, 1, 2)) = 0.45444282400422
    input(Array(1, 1, 3)) = 0.68826303933747
    input(Array(1, 2, 1)) = 0.95363974059001
    input(Array(1, 2, 2)) = 0.62628222838975
    input(Array(1, 2, 3)) = 0.60546887200326
    input(Array(1, 3, 1)) = 0.64541592891328
    input(Array(1, 3, 2)) = 0.89987210533582
    input(Array(1, 3, 3)) = 0.48417483456433
    input(Array(2, 1, 1)) = 0.202674907865
    input(Array(2, 1, 2)) = 0.092700937297195
    input(Array(2, 1, 3)) = 0.7684360188432
    input(Array(2, 2, 1)) = 0.54550718679093
    input(Array(2, 2, 2)) = 0.32182544097304
    input(Array(2, 2, 3)) = 0.223418089794
    input(Array(2, 3, 1)) = 0.28639033297077
    input(Array(2, 3, 2)) = 0.80424255039543
    input(Array(2, 3, 3)) = 0.10117407562211
    val gradOutput = Tensor[Double](2, 1, 2)
    gradOutput(Array(1, 1, 1)) = 0.60744401183911
    gradOutput(Array(1, 1, 2)) = 0.74274225486442
    gradOutput(Array(2, 1, 1)) = 0.071322691626847
    gradOutput(Array(2, 1, 2)) = 0.6038177870214
    val expectedOutput = Tensor[Double](2, 1, 2)
    expectedOutput(Array(1, 1, 1)) = 0.69643987486294
    expectedOutput(Array(1, 1, 2)) = 0.62641731727247
    expectedOutput(Array(2, 1, 1)) = 0.37555689271539
    expectedOutput(Array(2, 1, 2)) = 0.38529951882083
    val expectedGrad = Tensor[Double](2, 3, 3)
    expectedGrad(Array(1, 1, 1)) = 0.10124066863985
    expectedGrad(Array(1, 1, 2)) = 0.22503104445059
    expectedGrad(Array(1, 1, 3)) = 0.12379037581074
    expectedGrad(Array(1, 2, 1)) = 0.10124066863985
    expectedGrad(Array(1, 2, 2)) = 0.22503104445059
    expectedGrad(Array(1, 2, 3)) = 0.12379037581074
    expectedGrad(Array(1, 3, 1)) = 0.10124066863985
    expectedGrad(Array(1, 3, 2)) = 0.22503104445059
    expectedGrad(Array(1, 3, 3)) = 0.12379037581074
    expectedGrad(Array(2, 1, 1)) = 0.011887115271141
    expectedGrad(Array(2, 1, 2)) = 0.11252341310804
    expectedGrad(Array(2, 1, 3)) = 0.1006362978369
    expectedGrad(Array(2, 2, 1)) = 0.011887115271141
    expectedGrad(Array(2, 2, 2)) = 0.11252341310804
    expectedGrad(Array(2, 2, 3)) = 0.1006362978369
    expectedGrad(Array(2, 3, 1)) = 0.011887115271141
    expectedGrad(Array(2, 3, 2)) = 0.11252341310804
    expectedGrad(Array(2, 3, 3)) = 0.1006362978369
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

  "A SpatialAveragePooling in ceil mode" should "generate correct output and gradInput" in {
    val module = new SpatialAveragePooling[Double](3, 1, 2, 2).ceil()
    val input = Tensor[Double](1, 4, 3)
    input(Array(1, 1, 1)) = 0.25434372201562
    input(Array(1, 1, 2)) = 0.20443214406259
    input(Array(1, 1, 3)) = 0.33442943682894
    input(Array(1, 2, 1)) = 0.051310112234205
    input(Array(1, 2, 2)) = 0.56103343307041
    input(Array(1, 2, 3)) = 0.041837680386379
    input(Array(1, 3, 1)) = 0.75616162386723
    input(Array(1, 3, 2)) = 0.35945181339048
    input(Array(1, 3, 3)) = 0.4502888196148
    input(Array(1, 4, 1)) = 0.14862711215392
    input(Array(1, 4, 2)) = 0.050680571002886
    input(Array(1, 4, 3)) = 0.93014938035049
    val gradOutput = Tensor[Double](1, 3, 1)
    gradOutput(Array(1, 1, 1)) = 0.22147525195032
    gradOutput(Array(1, 2, 1)) = 0.30394183006138
    gradOutput(Array(1, 3, 1)) = 0.77438542619348
    val output = module.forward(input)
    output.size(1) should be(1)
    output.size(2) should be(3)
    output.size(3) should be(1)
    val expectedOutput = Tensor[Double](1, 3, 1)
    expectedOutput(Array(1, 1, 1)) = 0.264402
    expectedOutput(Array(1, 2, 1)) = 0.521967
    val gradInput = module.backward(input, gradOutput)
    val expectedGrad = Tensor[Double](1, 4, 3)
    expectedGrad(Array(1, 1, 1)) = 0.07382508398344
    expectedGrad(Array(1, 1, 2)) = 0.07382508398344
    expectedGrad(Array(1, 1, 3)) = 0.07382508398344
    expectedGrad(Array(1, 2, 1)) = 0.0
    expectedGrad(Array(1, 2, 2)) = 0.0
    expectedGrad(Array(1, 2, 3)) = 0.0
    expectedGrad(Array(1, 3, 1)) = 0.10131394335379333
    expectedGrad(Array(1, 3, 2)) = 0.10131394335379333
    expectedGrad(Array(1, 3, 3)) = 0.10131394335379333
    expectedGrad(Array(1, 4, 1)) = 0.0
    expectedGrad(Array(1, 4, 2)) = 0.0
    expectedGrad(Array(1, 4, 3)) = 0.0

    gradInput should be(expectedGrad)
  }

  "A SpatialAvgPooling of float" should "be good in gradient checker" in {
    val module = new SpatialAveragePooling[Float](2, 2)
    val input = Tensor[Float](1, 3, 3).rand()
    val checker = new GradientChecker(1e-4, 1e-2)
    checker.checkLayer[Float](module, input) should be(true)
  }

  "A SpatialAvgPooling with globalpooling" should "work properly" in {
    val module = new SpatialAveragePooling[Float](2, 2, globalPooling = true)
    val input = Tensor[Float](1, 3, 3).rand()
    val module2 = new SpatialAveragePooling[Float](3, 3)
    module.forward(input) should be (module2.forward(input))
  }

  "A SpatialAveragePooling" should "work with SAME padding using NCHW format" in {
    import tensor.TensorNumericMath.TensorNumeric.NumericFloat

    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = -1
    val padH = -1
    val layer = new SpatialAveragePooling(kW, kH, dW, dH, padW, padH, countIncludePad = false)

    val inputData = Array(
      1.0f, 2, 3, 4
    )

    val input = Tensor(Storage(inputData), 1, Array(1, 2, 2))
    val output = layer.updateOutput(input)
    val gradInput = layer.backward(input, output)
    output.storage().array() should be (Array(2.5f, 3, 3.5, 4))
    gradInput.storage().array() should be (Array(0.625f, 2.125, 2.375, 7.875))
  }

  "A SpatialAveragePooling with NHWC format" should "generate correct output and gradInput" in {

    import tensor.TensorNumericMath.TensorNumeric.NumericDouble
    case class Pooling(kW: Int, kH: Int, dW: Int, dH: Int, pW: Int, pH: Int)

    val params = for (kernel <- 1 to 5;
                      stride <- 1 to 5;
                      padding <- -1 to kernel / 2) yield {
      Pooling(kernel, kernel, stride, stride, padding, padding)
    }

    for (param <- params) {
      println(param)

      val module = new SpatialAveragePooling(param.kW, param.kH, param.dW, param.dH,
        param.pW, param.pH)
      val moduleNHWC = new SpatialAveragePooling(param.kW, param.kH, param.dW, param.dH,
        param.pW, param.pH, format = DataFormat.NHWC)

      val input = Tensor(2, 4, 10, 10).randn()

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

class SpatialAveragePoolingSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val spatialAveragePooling = new SpatialAveragePooling[Float](3, 2, 2, 1).
      setName("spatialAveragePooling")
    val input = Tensor[Float](1, 4, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(spatialAveragePooling, input)
  }
}
