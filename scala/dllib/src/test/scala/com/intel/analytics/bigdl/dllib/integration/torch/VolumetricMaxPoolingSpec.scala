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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{GradientChecker, VolumetricMaxPooling}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class VolumetricMaxPoolingSpec extends TorchSpec {

  "VolumetricMaxPooling Forward dim 4 Double" should "work properly" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 4).toInt
    val to = RNG.uniform(1, 4).toInt
    val kt = RNG.uniform(2, 4).toInt
    val ki = RNG.uniform(2, 4).toInt
    val kj = RNG.uniform(2, 4).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padW = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padH = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val batch = RNG.uniform(2, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = VolumetricMaxPooling[Double](kt, ki, kj, st, si, sj,
      padT, padW, padH)

    val input = Tensor[Double](from, int, ini, inj).apply1(e => Random.nextDouble())
    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"layer = nn.VolumetricMaxPooling($kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH)\n" +
      "output = layer:forward(input)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("output"))

    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]

    output.size() should be(luaOutput.size())
    output should be(luaOutput)
  }

  "VolumetricMaxPooling Forward dim 5 Double" should "work properly" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 4).toInt
    val to = RNG.uniform(1, 4).toInt
    val kt = RNG.uniform(2, 4).toInt
    val ki = RNG.uniform(2, 4).toInt
    val kj = RNG.uniform(2, 4).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padW = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padH = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val batch = RNG.uniform(2, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = VolumetricMaxPooling[Double](kt, ki, kj, st, si, sj,
      padT, padW, padH)

    val input = Tensor[Double](batch, from, int, ini, inj).apply1(e => Random.nextDouble())

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"layer = nn.VolumetricMaxPooling($kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH)\n" +
      "output = layer:forward(input)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("weight", "bias", "output"))

    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]

    output should be(luaOutput)
  }

  "forward backward double batch" should "work properly" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val from = RNG.uniform(3, 4).toInt
    val kt = RNG.uniform(2, 7).toInt
    val ki = RNG.uniform(2, 7).toInt
    val kj = RNG.uniform(2, 7).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padW = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padH = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val batch = RNG.uniform(2, 7).toInt

    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val module = VolumetricMaxPooling[Double](kt, ki, kj, st, si, sj,
      padT, padW, padH)

    Random.setSeed(seed)
    val input = Tensor[Double](batch, from, int, ini, inj).apply1(e => Random.nextDouble())

    val output = module.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"module = nn.VolumetricMaxPooling($kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input, gradOutput)"


    val gradOutput = Tensor[Double]().resizeAs(output).rand()
    val gradInput = module.backward(input, gradOutput)
    val (luaTime, torchResult) = TH.run(code, Map("input" -> input,
      "gradOutput" -> gradOutput), Array("output", "gradInput"))

    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    output should be(luaOutput)
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    gradInput should be(luaGradInput)
  }

  "gradient check double batch" should "work properly" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val from = RNG.uniform(2, 4).toInt
    val kt = RNG.uniform(2, 7).toInt
    val ki = RNG.uniform(2, 7).toInt
    val kj = RNG.uniform(2, 7).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padW = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padH = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val batch = RNG.uniform(2, 7).toInt

    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val module = VolumetricMaxPooling[Double](kt, ki, kj, st, si, sj,
      padT, padW, padH)

    Random.setSeed(seed)
    val input = Tensor[Double](batch, from, int, ini, inj).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-5)
    checker.checkLayer[Double](module, input, 1e-3) should be (true)
  }

  "forward backward double batch ceil" should "work properly" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val from = RNG.uniform(3, 4).toInt
    val kt = RNG.uniform(2, 5).toInt
    val ki = RNG.uniform(2, 5).toInt
    val kj = RNG.uniform(2, 5).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padW = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padH = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val batch = RNG.uniform(2, 7).toInt

    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val module = VolumetricMaxPooling[Double](kt, ki, kj, st, si, sj,
      padT, padW, padH).ceil()

    Random.setSeed(seed)
    val input = Tensor[Double](batch, from, int, ini, inj).apply1(e => Random.nextDouble())

    val output = module.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"module = nn.VolumetricMaxPooling($kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH):ceil()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input, gradOutput)"


    val gradOutput = Tensor[Double]().resizeAs(output).rand()
    val gradInput = module.backward(input, gradOutput)
    val (luaTime, torchResult) = TH.run(code, Map("input" -> input,
      "gradOutput" -> gradOutput), Array("output", "gradInput"))

    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    output should be(luaOutput)
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    gradInput should be(luaGradInput)
  }

  "VolumetricMaxPooling Forward dim 4 Float" should "work properly" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 4).toInt
    val to = RNG.uniform(1, 4).toInt
    val kt = RNG.uniform(2, 4).toInt
    val ki = RNG.uniform(2, 4).toInt
    val kj = RNG.uniform(2, 4).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padW = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padH = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val batch = RNG.uniform(2, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = VolumetricMaxPooling[Float](kt, ki, kj, st, si, sj,
      padT, padW, padH)

    val input = Tensor[Float](from, int, ini, inj).apply1(e => Random.nextFloat())
    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "torch.setdefaulttensortype('torch.FloatTensor')" +
      s"layer = nn.VolumetricMaxPooling($kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH)\n" +
      "output = layer:forward(input)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("output"))

    val luaOutput = torchResult("output").asInstanceOf[Tensor[Float]]

    output.size() should be(luaOutput.size())
    output should be(luaOutput)
  }

  "VolumetricMaxPooling Forward dim 5 Float" should "work properly" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 4).toInt
    val to = RNG.uniform(1, 4).toInt
    val kt = RNG.uniform(2, 4).toInt
    val ki = RNG.uniform(2, 4).toInt
    val kj = RNG.uniform(2, 4).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padW = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padH = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val batch = RNG.uniform(2, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = VolumetricMaxPooling[Float](kt, ki, kj, st, si, sj,
      padT, padW, padH)

    val input = Tensor[Float](batch, from, int, ini, inj).apply1(e => Random.nextFloat())

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "torch.setdefaulttensortype('torch.FloatTensor')" +
      s"layer = nn.VolumetricMaxPooling($kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH)\n" +
      "output = layer:forward(input)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("weight", "bias", "output"))

    val luaOutput = torchResult("output").asInstanceOf[Tensor[Float]]

    output should be(luaOutput)
  }


  "forward backward Float batch" should "work properly" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val from = RNG.uniform(3, 5).toInt
    val kt = RNG.uniform(2, 7).toInt
    val ki = RNG.uniform(2, 7).toInt
    val kj = RNG.uniform(2, 7).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padW = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padH = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val batch = RNG.uniform(2, 7).toInt

    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val module = VolumetricMaxPooling[Float](kt, ki, kj, st, si, sj,
      padT, padW, padH)

    Random.setSeed(seed)
    val input = Tensor[Float](batch, from, int, ini, inj).apply1(e => Random.nextFloat())

    val output = module.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "torch.setdefaulttensortype('torch.FloatTensor')" +
      s"module = nn.VolumetricMaxPooling($kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input, gradOutput)"


    val gradOutput = Tensor[Float]()
    gradOutput.resizeAs(output).rand()
    val gradInput = module.backward(input, gradOutput)
    val (luaTime, torchResult) = TH.run(code, Map("input" -> input,
      "gradOutput" -> gradOutput), Array("output", "gradInput"))

    val luaOutput = torchResult("output").asInstanceOf[Tensor[Float]]
    output should be(luaOutput)
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    gradInput should be(luaGradInput)
  }

  "forward backward Float batch ceil" should "work properly" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val from = RNG.uniform(3, 5).toInt
    val kt = RNG.uniform(2, 5).toInt
    val ki = RNG.uniform(2, 5).toInt
    val kj = RNG.uniform(2, 5).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padW = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val padH = Math.min(RNG.uniform(0, 2).toInt, math.floor(kt / 2).toInt)
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val batch = RNG.uniform(2, 7).toInt

    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val module = VolumetricMaxPooling[Float](kt, ki, kj, st, si, sj,
      padT, padW, padH).ceil()

    Random.setSeed(seed)
    val input = Tensor[Float](batch, from, int, ini, inj).apply1(e => Random.nextFloat())

    val output = module.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "torch.setdefaulttensortype('torch.FloatTensor')" +
      s"module = nn.VolumetricMaxPooling($kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH):ceil()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input, gradOutput)"


    val gradOutput = Tensor[Float]().resizeAs(output).rand()
    val gradInput = module.backward(input, gradOutput)
    val (luaTime, torchResult) = TH.run(code, Map("input" -> input,
      "gradOutput" -> gradOutput), Array("output", "gradInput"))

    val luaOutput = torchResult("output").asInstanceOf[Tensor[Float]]
    output should be(luaOutput)
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Float]]
    gradInput should be(luaGradInput)
  }
}
