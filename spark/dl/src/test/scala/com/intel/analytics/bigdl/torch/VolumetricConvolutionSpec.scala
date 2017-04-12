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

package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class VolumetricConvolutionSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A VolumetricConvolution" should "generate correct output" in {
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 4).toInt
    val to = RNG.uniform(1, 4).toInt
    val kt = RNG.uniform(1, 4).toInt
    val ki = RNG.uniform(1, 4).toInt
    val kj = RNG.uniform(1, 4).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = RNG.uniform(0, 2).toInt
    val padW = RNG.uniform(0, 2).toInt
    val padH = RNG.uniform(0, 2).toInt
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH)

    Random.setSeed(seed)
    val input = Tensor[Double](3, 100, 56, 56).apply1(e => Random.nextDouble())

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"layer = nn.VolumetricConvolution($from, $to, $kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH)\n" +
      "weight = layer.weight\n" +
      "bias = layer.bias \n" +
      "output = layer:forward(input)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("weight", "bias", "output"))

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be equals luaWeight
    bias should be equals luaBias
    output should be equals luaOutput
  }

  "A VolumetricConvolution without bias" should "generate correct output" in {
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 4).toInt
    val to = RNG.uniform(1, 4).toInt
    val kt = RNG.uniform(1, 4).toInt
    val ki = RNG.uniform(1, 4).toInt
    val kj = RNG.uniform(1, 4).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = RNG.uniform(0, 2).toInt
    val padW = RNG.uniform(0, 2).toInt
    val padH = RNG.uniform(0, 2).toInt
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH, withBias = false)

    Random.setSeed(seed)
    val input = Tensor[Double](3, 100, 56, 56).apply1(e => Random.nextDouble())

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"layer = nn.VolumetricConvolution($from, $to, $kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH):noBias()\n" +
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

    weight should be equals luaWeight
    bias should be equals luaBias
    output should be equals luaOutput
  }


  "A VolumetricConvolution with batch input" should "generate correct output" in {
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 10).toInt
    val to = RNG.uniform(1, 10).toInt
    val kt = RNG.uniform(1, 6).toInt
    val ki = RNG.uniform(1, 6).toInt
    val kj = RNG.uniform(1, 6).toInt
    val st = RNG.uniform(1, 7).toInt
    val si = RNG.uniform(1, 7).toInt
    val sj = RNG.uniform(1, 7).toInt
    val padT = RNG.uniform(0, 2).toInt
    val padW = RNG.uniform(0, 2).toInt
    val padH = RNG.uniform(0, 2).toInt
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val batch = RNG.uniform(2, 10).toInt
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH)

    Random.setSeed(seed)
    val input = Tensor[Double](batch, from, int, inj, ini).apply1(e => Random.nextDouble())

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"layer = nn.VolumetricConvolution($from, $to, $kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH)\n" +
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

    weight should be equals luaWeight
    bias should be equals luaBias
    output should be equals luaOutput
  }

  "A VolumetricConvolution with batch input no bias" should "generate correct output" in {
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 10).toInt
    val to = RNG.uniform(1, 10).toInt
    val kt = RNG.uniform(1, 6).toInt
    val ki = RNG.uniform(1, 6).toInt
    val kj = RNG.uniform(1, 6).toInt
    val st = RNG.uniform(1, 7).toInt
    val si = RNG.uniform(1, 7).toInt
    val sj = RNG.uniform(1, 7).toInt
    val padT = RNG.uniform(0, 2).toInt
    val padW = RNG.uniform(0, 2).toInt
    val padH = RNG.uniform(0, 2).toInt
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val batch = RNG.uniform(2, 10).toInt
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH, withBias = false)

    Random.setSeed(seed)
    val input = Tensor[Double](batch, from, int, inj, ini).apply1(e => Random.nextDouble())

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"layer = nn.VolumetricConvolution($from, $to, $kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH):noBias()\n" +
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

    weight should be equals luaWeight
    bias should be equals luaBias
    output should be equals luaOutput
  }

  "A VolumetricConvolution" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 4).toInt
    val to = RNG.uniform(1, 4).toInt
    val kt = RNG.uniform(1, 4).toInt
    val ki = RNG.uniform(1, 4).toInt
    val kj = RNG.uniform(1, 4).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = RNG.uniform(0, 2).toInt
    val padW = RNG.uniform(0, 2).toInt
    val padH = RNG.uniform(0, 2).toInt
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH)

    val input = Tensor[Double](from, int, ini, inj).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer[Double](layer, input, 1e-3) should be(true)
  }

  "A VolumetricConvolution with batch" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 6).toInt
    val to = RNG.uniform(1, 4).toInt
    val kt = RNG.uniform(1, 4).toInt
    val ki = RNG.uniform(1, 4).toInt
    val kj = RNG.uniform(1, 4).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = RNG.uniform(0, 2).toInt
    val padW = RNG.uniform(0, 2).toInt
    val padH = RNG.uniform(0, 2).toInt
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val batch = RNG.uniform(2, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH)

    val input = Tensor[Double](batch, from, int, ini, inj).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer[Double](layer, input, 1e-3) should be(true)
  }

  "A VolumetricConvolution" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 4).toInt
    val to = RNG.uniform(1, 4).toInt
    val kt = RNG.uniform(1, 4).toInt
    val ki = RNG.uniform(1, 4).toInt
    val kj = RNG.uniform(1, 4).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = RNG.uniform(0, 2).toInt
    val padW = RNG.uniform(0, 2).toInt
    val padH = RNG.uniform(0, 2).toInt
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH)

    val input = Tensor[Double](from, int, ini, inj).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkWeight[Double](layer, input, 1e-3) should be(true)
  }

  "A VolumetricConvolution with batch" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 6).toInt
    val to = RNG.uniform(1, 4).toInt
    val kt = RNG.uniform(1, 4).toInt
    val ki = RNG.uniform(1, 4).toInt
    val kj = RNG.uniform(1, 4).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = RNG.uniform(0, 2).toInt
    val padW = RNG.uniform(0, 2).toInt
    val padH = RNG.uniform(0, 2).toInt
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val batch = RNG.uniform(2, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH)

    val input = Tensor[Double](batch, from, int, ini, inj).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkWeight[Double](layer, input, 1e-3) should be(true)
  }
}

