/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.feature.image3d

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.zoo.feature.{TH, TorchSpec}


class RotationTransformerSpec extends TorchSpec {
  "A RotationTransformer" should "generate correct output when dimension of depth is 1" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](1, 10, 10)
    input.apply1(e => RNG.uniform(0, 1).toFloat)
    input.resize(1, 10, 10, 1)
    val rotAngles = Array[Double](0, 0, math.Pi/3.7)
    val rot = Rotate3D(rotAngles)
    val image = ImageFeature3D(input)
    val dst = rot.transform(image)
    val code = "require 'image'\n" +
    "dst = image.rotate(src,math.pi/3.7,'bilinear')"
    val (luaTime, torchResult) = TH.run(code,
      Map("src" -> input.view(10, 10)),
      Array("dst"))
    val dstTorch = torchResult("dst").asInstanceOf[Tensor[Float]]
    dst[Tensor[Float]](ImageFeature.imageTensor).view(10, 10).map(dstTorch, (v1, v2) => {
      assert(math.abs(v1-v2)<1e-6)
      v1
    })
  }// end test

  "A RotationTransformer" should "generate correct output when dimension of height is 1" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](10, 1, 10)
    input.apply1(e => (RNG.uniform(0, 1).toFloat))
    input.resize(10, 1, 10, 1)
    val rotAngles = Array[Double](math.Pi/3.7, 0, 0)
    val rot = Rotate3D(rotAngles)
    val image = ImageFeature3D(input)
    val dst = rot.transform(image)
    // The z-axis is pointing downward so the rotation direction is opposite to normal one.
    val code = "require 'image'\n" +
    "dst = image.rotate(src,-math.pi/3.7,'bilinear')"
    val (luaTime, torchResult) = TH.run(code,
      Map("src" -> input.view(10, 10)),
      Array("dst"))
    val dstTorch = torchResult("dst").asInstanceOf[Tensor[Float]]
    dst[Tensor[Float]](ImageFeature.imageTensor).view(10, 10).map(dstTorch, (v1, v2) => {
      assert(math.abs(v1-v2)<1e-6)
      v1
    })
  }// end test

  "A RotationTransformer" should "generate correct output when dimension of width is 1" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](10, 10, 1)
    input.apply1(e => (RNG.uniform(0, 1).toFloat))
    input.resize(10, 10, 1, 1)
    val rotAngles = Array[Double](0, math.Pi/3.7, 0)
    val rot = Rotate3D(rotAngles)
    val image = ImageFeature3D(input)
    val dst = rot.transform(image)
    val code = "require 'image'\n" +
    "dst = image.rotate(src,math.pi/3.7,'bilinear')"
    val (luaTime, torchResult) = TH.run(code,
      Map("src" -> input.view(10, 10)),
      Array("dst"))
    val dstTorch = torchResult("dst").asInstanceOf[Tensor[Float]]
    dst[Tensor[Float]](ImageFeature.imageTensor).view(10, 10).map(dstTorch, (v1, v2) => {
      assert(math.abs(v1-v2)<1e-6)
      v1
    })

  }
}
