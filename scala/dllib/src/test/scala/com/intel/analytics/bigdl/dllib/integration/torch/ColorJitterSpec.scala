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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._
import scala.util.Random
import com.intel.analytics.bigdl.dataset.LocalArrayDataSet
import com.intel.analytics.bigdl.dataset.image.{ColorJitter, LabeledBGRImage}
import com.intel.analytics.bigdl.utils.RandomGenerator

@com.intel.analytics.bigdl.tags.Serial
class ColorJitterSpec extends TorchSpec {
  "A ColorJitter" should "blend image correctly" in {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }

    val seed = 1000
    RNG.setSeed(seed)
    val image1 = new LabeledBGRImage((1 to 27).map(_.toFloat).toArray, 3, 3, 0)
    val image2 = new LabeledBGRImage((2 to 28).map(_.toFloat).toArray, 3, 3, 0)
    val image3 = new LabeledBGRImage((3 to 29).map(_.toFloat).toArray, 3, 3, 0)
    val expected = image1.clone()
    val labeledBGRImage =
      new LabeledBGRImage(image1.content, image1.width(), image1.height(), image1.label())
    val colorJitter = ColorJitter()
    val iter = colorJitter.apply(Iterator.single(labeledBGRImage))
    val test = iter.next()

    val torchInput =
      Tensor[Float](Storage(expected.content), storageOffset = 1, size = Array(3, 3, 3))
        .transpose(1, 3).transpose(2, 3)
    println(s"torchInput = ${torchInput}")
    val code =
      "torch.setdefaulttensortype('torch.FloatTensor')" +
        "torch.manualSeed(" + seed + ")\n" +
      """
        |local function blend(img1, img2, alpha)
        |   return img1:mul(alpha):add(1 - alpha, img2)
        |end
        |
        |local function grayscale(dst, img)
        |   dst:resizeAs(img)
        |   dst[1]:zero()
        |   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
        |   dst[2]:copy(dst[1])
        |   dst[3]:copy(dst[1])
        |   return dst
        |end
        |
        |function Saturation(var)
        |   local gs
        |
        |   return function(input)
        |      gs = gs or input.new()
        |      grayscale(gs, input)
        |      local alpha = 1.0 + torch.uniform(-var, var)
        |      blend(input, gs, alpha)
        |      return input
        |   end
        |end
        |
        |function Brightness(var)
        |   local gs
        |
        |   return function(input)
        |      gs = gs or input.new()
        |      gs:resizeAs(input):zero()
        |
        |      local alpha = 1.0 + torch.uniform(-var, var)
        |      blend(input, gs, alpha)
        |      return input
        |   end
        |end
        |
        |function Contrast(var)
        |   local gs
        |
        |   return function(input)
        |      gs = gs or input.new()
        |      grayscale(gs, input)
        |      gs:fill(gs[1]:mean())
        |      local alpha = 1.0 + torch.uniform(-var, var)
        |      --local alpha = 0.61087716
        |      blend(input, gs, alpha)
        |      return input
        |   end
        |end
        |
        |function RandomOrder(ts)
        |   return function(input)
        |      local img = input.img or input
        |      local order = torch.randperm(#ts)
        |      for i=1,#ts do
        |         img = ts[order[i]](img)
        |      end
        |      return img
        |   end
        |end
        |
        |function ColorJitter(opt)
        |   local brightness = opt.brightness or 0
        |   local contrast = opt.contrast or 0
        |   local saturation = opt.saturation or 0
        |
        |   local ts = {}
        |   if brightness ~= 0 then
        |      table.insert(ts, Brightness(brightness))
        |   end
        |   if contrast ~= 0 then
        |      table.insert(ts, Contrast(contrast))
        |   end
        |   if saturation ~= 0 then
        |      table.insert(ts, Saturation(saturation))
        |   end
        |
        |   if #ts == 0 then
        |      return function(input) return input end
        |   end
        |
        |   return RandomOrder(ts)
        |end
        |
        |local transform = ColorJitter({
        |            brightness = 0.4,
        |            contrast = 0.4,
        |            saturation = 0.4,
        |})
        |output = transform(input)
      """.stripMargin

    val (luaTime, torchResult) = TH.run(code, Map("input" -> torchInput), Array("output"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Float]]

    val bigdlOutput = Tensor[Float](Storage(test.content), storageOffset = 1, size = Array(3, 3, 3))
      .transpose(1, 3).transpose(2, 3)
    luaOutput.map(bigdlOutput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-5)
      v1
    })
  }
}
