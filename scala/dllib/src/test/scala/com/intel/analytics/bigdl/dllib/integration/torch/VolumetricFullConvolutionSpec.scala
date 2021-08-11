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

package com.intel.analytics.bigdl.integration.torch

import com.intel.analytics.bigdl.nn.{BilinearFiller, Sequential, VolumetricFullConvolution, Zeros}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class VolumetricFullConvolutionSpec extends TorchSpec {
  "A VolumetricFullConvolution" should "generate correct output" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 3
    val nOutputPlane = 6
    val kT = 4
    val kW = 3
    val kH = 3
    val dT = 2
    val dW = 1
    val dH = 1
    val padT = 2
    val padW = 2
    val padH = 2
    val layer = new VolumetricFullConvolution[Double](nInputPlane, nOutputPlane,
      kT, kW, kH, dT, dW, dH, padT, padW, padH)

    Random.setSeed(seed)
    val input = Tensor[Double](3, 3, 3, 6, 6).apply1(e => Random.nextDouble())
    layer.updateOutput(input)
    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"layer = nn.VolumetricFullConvolution($nInputPlane," +
      s" $nOutputPlane, $kT, $kW, $kH, $dT, $dW, $dH, $padT, $padW, $padH)\n" +
      "weight = layer.weight\n" +
      "bias = layer.bias \n" +
      "output = layer:forward(input) "

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("weight", "bias", "output"))

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be(luaWeight.resizeAs(weight))
    bias should be(luaBias)
    output should be(luaOutput)
  }

  "A VolumetricFullConvolution on rectangle input" should "generate correct output" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 10
    val nOutputPlane = 10
    val kT = 5
    val kW = 4
    val kH = 4
    val dT = 3
    val dW = 2
    val dH = 2
    val padT = 2
    val padW = 1
    val padH = 1
    val layer = new VolumetricFullConvolution[Double](nInputPlane, nOutputPlane,
      kT, kW, kH, dT, dW, dH, padT, padW, padH)

    Random.setSeed(seed)
    val input = Tensor[Double](1, nInputPlane, 10, 20, 30).apply1(e => Random.nextDouble())
    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"layer = nn.VolumetricFullConvolution($nInputPlane," +
      s" $nOutputPlane, $kT, $kW, $kH, $dT, $dW, $dH, $padT, $padW, $padH)\n" +
      "weight = layer.weight\n" +
      "bias = layer.bias \n" +
      "output = layer:forward(input) "

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("weight", "bias", "output"))

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be(luaWeight.resizeAs(weight))
    bias should be(luaBias)
    output should be(luaOutput)
  }

  "A VolumetricFullConvolution" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 3
    val nOutputPlane = 6
    val kT = 4
    val kW = 3
    val kH = 3
    val dT = 2
    val dW = 1
    val dH = 1
    val padT = 2
    val padW = 2
    val padH = 2
    val layer = new VolumetricFullConvolution[Double](nInputPlane, nOutputPlane,
      kT, kW, kH, dT, dW, dH, padT, padW, padH)
    val model = new Sequential[Double]()
    model.add(layer)

    Random.setSeed(3)
    val input = Tensor[Double](3, nInputPlane, 3, 6, 6).apply1(e => Random.nextDouble())
    var output = model.updateOutput(input).toTensor[Double]

    val gradOutput = Tensor[Double]().resizeAs(output).apply1(e => Random.nextDouble())

    var gradInput = model.backward(input, gradOutput)
    output = model.updateOutput(input).toTensor[Double]
    gradInput = model.backward(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"""layer = nn.VolumetricFullConvolution($nInputPlane,
          $nOutputPlane, $kT, $kW, $kH, $dT, $dW, $dH, $padT, $padW, $padH)
      model = nn.Sequential()
      model:add(layer)
      weight = layer.weight
      bias = layer.bias
      model:zeroGradParameters()
      output = model:forward(input)
      gradInput = model:backward(input, gradOutput)
      output = model:forward(input)
      gradInput = model:backward(input, gradOutput)
      gradBias = layer.gradBias
      gradWeight = layer.gradWeight
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput),
      Array("weight", "bias", "output", "gradInput", "gradBias", "gradWeight")
    )

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]
    val luaGradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be(luaWeight.resizeAs(weight))
    bias should be(luaBias)
    output should be(luaOutput)
    gradInput should be(luaGradInput)
    luaGradBias should be (layer.gradBias)
    luaGradWeight should be (layer.gradWeight.resizeAs(luaGradWeight))
  }

  "A VolumetricFullConvolution" should "generate correct output and grad with 4D input" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 3
    val nOutputPlane = 6
    val kT = 4
    val kW = 3
    val kH = 3
    val dT = 2
    val dW = 1
    val dH = 1
    val padT = 2
    val padW = 2
    val padH = 2
    val layer = new VolumetricFullConvolution[Double](nInputPlane, nOutputPlane,
      kT, kW, kH, dT, dW, dH, padT, padW, padH)
    val model = new Sequential[Double]()
    model.add(layer)

    Random.setSeed(3)
    val input = Tensor[Double](nInputPlane, 3, 6, 6).apply1(e => Random.nextDouble())
    val output = model.updateOutput(input).toTensor[Double]

    val gradOutput = Tensor[Double]().resizeAs(output).apply1(e => Random.nextDouble())

    val gradInput = model.backward(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"""layer = nn.VolumetricFullConvolution($nInputPlane,
          $nOutputPlane, $kT, $kW, $kH, $dT, $dW, $dH, $padT, $padW, $padH)
      model = nn.Sequential()
      model:add(layer)
      weight = layer.weight
      bias = layer.bias
      model:zeroGradParameters()
      output = model:forward(input)
      gradInput = model:backward(input, gradOutput)
      gradBias = layer.gradBias
      gradWeight = layer.gradWeight
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput),
      Array("weight", "bias", "output", "gradInput", "gradBias", "gradWeight")
    )

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]
    val luaGradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be(luaWeight.resizeAs(weight))
    bias should be(luaBias)
    output should be(luaOutput)
    gradInput should be(luaGradInput)
    luaGradBias should be (layer.gradBias)
    luaGradWeight should be (layer.gradWeight.resizeAs(luaGradWeight))
  }

  "A VolumetricFullConvolution noBias" should "generate correct output and grad with 3D input" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 3
    val nOutputPlane = 6
    val kT = 4
    val kW = 3
    val kH = 3
    val dT = 2
    val dW = 1
    val dH = 1
    val padT = 2
    val padW = 2
    val padH = 2
    val layer = new VolumetricFullConvolution[Double](nInputPlane, nOutputPlane,
      kT, kW, kH, dT, dW, dH, padT, padW, padH, noBias = true)
    val model = new Sequential[Double]()
    model.add(layer)

    Random.setSeed(3)
    val input = Tensor[Double](3, nInputPlane, 6, 6).apply1(e => Random.nextDouble())
    val output = model.updateOutput(input).toTensor[Double]

    val gradOutput = Tensor[Double]().resizeAs(output).apply1(e => Random.nextDouble())

    val gradInput = model.backward(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"""layer = nn.VolumetricFullConvolution($nInputPlane,
          $nOutputPlane, $kT, $kW, $kH, $dT, $dW, $dH, $padT, $padW, $padH)
         layer:noBias()
      model = nn.Sequential()
      model:add(layer)
      weight = layer.weight
      bias = layer.bias
      model:zeroGradParameters()
      output = model:forward(input)
      gradInput = model:backward(input, gradOutput)
      gradBias = layer.gradBias
      gradWeight = layer.gradWeight
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput),
      Array("weight", "output", "gradInput", "gradWeight")
    )

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be(luaWeight.resizeAs(weight))
    output should be(luaOutput)
    gradInput should be(luaGradInput)
    luaGradWeight should be (layer.gradWeight.resizeAs(luaGradWeight))
  }

  "A VolumetricFullConvolution" should "generate correct output and grad with table input" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 3
    val nOutputPlane = 6
    val kT = 3
    val kW = 3
    val kH = 3
    val dT = 2
    val dW = 2
    val dH = 2
    val padW = 1
    val padT = 1
    val padH = 1
    val layer = new VolumetricFullConvolution[Double](nInputPlane, nOutputPlane,
      kT, kW, kH, dT, dW, dH, padT, padW, padH)

    Random.setSeed(3)
    val input1 = Tensor[Double](nInputPlane, 3, 6, 6).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](nInputPlane, 6, 6).apply1(e => Random.nextInt(dH))
    val input = T(input1, input2)
    val output = layer.updateOutput(input)

    val gradOutput = Tensor[Double]().resizeAs(output).apply1(e => Random.nextDouble())

    val gradInput = layer.backward(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"""layer = nn.VolumetricFullConvolution($nInputPlane,
          $nOutputPlane, $kT, $kW, $kH, $dT, $dW, $dH, $padT, $padW, $padH)
         input = {input1, input2}
      model = nn.Sequential()
      model:add(layer)
      weight = layer.weight
      bias = layer.bias
      model:zeroGradParameters()
      output = model:forward(input)
      gradInput = model:backward(input, gradOutput)
      gradBias = layer.gradBias
      gradWeight = layer.gradWeight
      gradInput1 = gradInput[1]
      gradInput2 = gradInput[2]
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input1" -> input1, "input2" -> input2, "gradOutput" -> gradOutput),
      Array("weight", "bias", "output", "gradInput1", "gradInput2", "gradBias", "gradWeight")
    )

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput1 = torchResult("gradInput1").asInstanceOf[Tensor[Double]]
    val luaGradInput2 = torchResult("gradInput2").asInstanceOf[Tensor[Double]]
    val luaGradInput = T(luaGradInput1, luaGradInput2)
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]
    val luaGradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be(luaWeight.resizeAs(weight))
    bias should be(luaBias)
    output should be(luaOutput)
    gradInput should be(luaGradInput)
    luaGradBias should be (layer.gradBias)
    luaGradWeight should be (layer.gradWeight.resizeAs(luaGradWeight))
  }

  "A VolumetricFullConvolution OneToOne" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 6
    val nOutputPlane = 6
    val kT = 3
    val kW = 3
    val kH = 3
    val dT = 1
    val dW = 1
    val dH = 1
    val layer = new VolumetricFullConvolution[Double](nInputPlane, nOutputPlane,
      kT, kW, kH, dT, dW, dH)
    Random.setSeed(3)
    val input = Tensor[Double](6, nInputPlane, 5, 5).apply1(e => Random.nextDouble())
    val output = layer.forward(input).toTensor[Double]

    val gradOutput = Tensor[Double]().resizeAs(output).apply1(e => Random.nextDouble())

    val gradInput = layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"""tt = nn.tables.oneToOne(6)
      layer = nn.VolumetricFullConvolution($nInputPlane,
      $nOutputPlane, $kT, $kW, $kH, $dT, $dW, $dH)
      layer.weight:copy(weight)
      layer.bias:copy(bias)
      model = nn.Sequential()
      model:add(layer)
      model:zeroGradParameters()
      output = model:forward(input)
      gradInput = model:backward(input, gradOutput)
      gradBias = layer.gradBias
      gradWeight = layer.gradWeight
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput,
        "weight" -> layer.weight, "bias" -> layer.bias),
      Array("output", "gradInput", "gradBias", "gradWeight")
    )

    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]
    val luaGradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]

    val weight = layer.weight
    val bias = layer.bias

    output should be(luaOutput)
    gradInput should be(luaGradInput)
    luaGradBias should be (layer.gradBias)
    luaGradWeight should be (layer.gradWeight.resizeAs(luaGradWeight))
  }

  "A VolumetricFullConvolution with different input" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 3
    val nOutputPlane = 6
    val kT = 4
    val kW = 3
    val kH = 3
    val dT = 2
    val dW = 1
    val dH = 1
    val padT = 2
    val padW = 2
    val padH = 2
    val layer = new VolumetricFullConvolution[Double](nInputPlane, nOutputPlane,
      kT, kW, kH, dT, dW, dH, padT, padW, padH)
    val model = new Sequential[Double]()
    model.add(layer)

    Random.setSeed(3)
    val input = Tensor[Double](3, nInputPlane, 3, 6, 6).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](6, nInputPlane, 3, 6, 6).apply1(e => Random.nextDouble())
    val output = model.updateOutput(input).toTensor[Double]
    val gradOutput = Tensor[Double]().resizeAs(output).apply1(e => Random.nextDouble())
    val gradInput = model.backward(input, gradOutput)

    model.zeroGradParameters()
    val output2 = model.updateOutput(input2).toTensor[Double]
    val gradOutput2 = Tensor[Double]().resizeAs(output2).apply1(e => Random.nextDouble())
    val gradInput2 = model.backward(input2, gradOutput2)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"""layer = nn.VolumetricFullConvolution($nInputPlane,
          $nOutputPlane, $kT, $kW, $kH, $dT, $dW, $dH, $padT, $padW, $padH)
      model = nn.Sequential()
      model:add(layer)
      weight = layer.weight
      bias = layer.bias
      model:zeroGradParameters()
      output = model:forward(input)
      gradInput = model:backward(input, gradOutput)
      gradBias = layer.gradBias
      gradWeight = layer.gradWeight
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input2, "gradOutput" -> gradOutput2),
      Array("weight", "bias", "output", "gradInput", "gradBias", "gradWeight")
    )

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]
    val luaGradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be(luaWeight.resizeAs(weight))
    bias should be(luaBias)
    output2 should be(luaOutput)
    gradInput2 should be(luaGradInput)
    luaGradBias should be (layer.gradBias)
    luaGradWeight should be (layer.gradWeight.resizeAs(luaGradWeight))
  }

}
