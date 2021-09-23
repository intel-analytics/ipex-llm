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
package com.intel.analytics.bigdl.dllib.feature.image3d

import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.dllib.utils.RandomGenerator._
import org.scalatest.{FlatSpec, Matchers}


class RotationTransformerSpec extends FlatSpec with Matchers{
  "A RotationTransformer" should "generate correct output when dimension of depth is 1" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](1, 10, 10)
    input.apply1(e => RNG.uniform(0, 1).toFloat)
    input.resize(1, 10, 10, 1)
    val rotAngles = Array[Double](0, 0, math.Pi/3.7)
    val rot = Rotate3D(rotAngles)
    val image = ImageFeature3D(input)
    val dst = rot.transform(image)
    val dstTorch = Tensor[Double](Array[Double](0.0, 0.0, 0.68720585, 0.4518023, 0.619756,
      0.26050177, 0.48498562, 0.0, 0.0, 0.0, 0.0, 0.5075366, 0.5610923, 0.22793348, 0.15701589,
      0.21100126, 0.28684008, 0.48019138, 0.5790498, 0.0, 0.0, 0.51299816, 0.5537319, 0.37157783,
      0.42960122, 0.4258391, 0.25461295, 0.07401075, 0.36288196, 0.46194813, 0.36622074,
      0.40474024, 0.42051914, 0.30529416, 0.35716558, 0.4104683, 0.3634557, 0.18230169,
      0.12811545, 0.5179936, 0.559532, 0.3976659, 0.35664213, 0.19423184, 0.7187436, 0.73311293,
      0.66162705, 0.40453663, 0.40114495, 0.41675943, 0.39625713, 0.39887276, 0.48166054,
      0.50793016, 0.56699604, 0.3960097, 0.4437459, 0.69167227, 0.5400801, 0.652137, 0.8049347,
      0.8948945, 0.86430603, 0.600158, 0.50343186, 0.20795111, 0.23471259, 0.39646378, 0.07110661,
      0.20663853, 0.95886403, 0.73128873, 0.5521641, 0.08182517, 0.4551149, 0.6015099, 0.36086103,
      0.44722062, 0.100247495, 0.0, 0.0, 0.42081234, 0.34027046, 0.4018491, 0.54632396, 0.86398995,
      0.27266297, 0.48549172, 0.21826318, 0.0, 0.0, 0.0, 0.0, 0.6506955, 0.27804166, 0.75620246,
      0.69736207, 0.8209634, 0.0, 0.0), Array(10, 10))
    val dstTensor = Tensor[Double](
      storage = Storage[Double](dst[Tensor[Float]](ImageFeature.imageTensor).storage().array()
        .map(_.toDouble)), storageOffset = 1, size = Array(1, 10, 10))
    dstTensor.view(10, 10).map(dstTorch, (v1, v2) => {
      assert(math.abs(v1-v2)<1e-6)
      v1
    })
  }// end test

  "A RotationTransformer" should "generate correct output when dimension of height is 1" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](10, 1, 10)
    input.apply1(e => (RNG.uniform(0, 1).toFloat))
    input.resize(10, 1, 10, 1)
    val rotAngles = Array[Double](math.Pi/3.7, 0, 0)
    val rot = Rotate3D(rotAngles)
    val image = ImageFeature3D(input)
    val dst = rot.transform(image)
    val dstTorch = Tensor[Double](Array[Double](0.0, 0.0, 0.0, 0.9337607, 0.63041264, 0.41022006,
      0.4238756, 0.38179016, 0.0, 0.0, 0.0, 0.3837239, 0.48049465, 0.86388755, 0.66710216,
      0.3300809, 0.28570667, 0.45415232, 0.44336346, 0.0, 0.66483647, 0.43097502, 0.3237357,
      0.8518211, 0.66504264, 0.34306353, 0.39141575, 0.7155649, 0.51030886, 0.0, 0.54608387,
      0.43258587, 0.25251946, 0.43319377, 0.5893668, 0.23969816, 0.25210613, 0.511659, 0.29608712,
      0.5754271, 0.57577354, 0.75738704, 0.5187364, 0.45187342, 0.5245649, 0.71314573, 0.19431172,
      0.33923146, 0.24373981, 0.69405204, 0.685539, 0.64075524, 0.5302063, 0.14410771, 0.43040138,
      0.7441464, 0.50390047, 0.5271677, 0.098411895, 0.42417723, 0.7500948, 0.34451854, 0.25963393,
      0.35118395, 0.45892385, 0.74234974, 0.34206071, 0.27863282, 0.26684713, 0.29670718,
      0.0, 0.30849406, 0.2423132, 0.51493, 0.63804555, 0.3836444, 0.09990372, 0.14387289,
      0.481886, 0.5588326, 0.0, 0.10782696, 0.14325471, 0.17757198, 0.5234246, 0.33868256,
      0.083803356, 0.45793825, 0.5710847, 0.0, 0.0, 0.0, 0.22376697, 0.33324337, 0.50276846,
      0.5293505, 0.43991444, 0.0, 0.0, 0.0), Array(10, 10))
    val dstTensor = Tensor[Double](
      storage = Storage[Double](dst[Tensor[Float]](ImageFeature.imageTensor).storage().array()
        .map(_.toDouble)), storageOffset = 1, size = Array(10, 1, 10))
    dstTensor.view(10, 10).map(dstTorch, (v1, v2) => {
      assert(math.abs(v1-v2)<1e-6)
      v1
    })
  }// end test

  "A RotationTransformer" should "generate correct output when dimension of width is 1" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](10, 10, 1)
    input.apply1(e => (RNG.uniform(0, 1).toFloat))
    input.resize(10, 10, 1, 1)
    val rotAngles = Array[Double](0, math.Pi/3.7, 0)
    val rot = Rotate3D(rotAngles)
    val image = ImageFeature3D(input)
    val dst = rot.transform(image)
    val dstTorch = Tensor[Double](Array[Double](0.0, 0.0, 0.68720585, 0.4518023, 0.619756,
      0.26050177, 0.48498562, 0.0, 0.0, 0.0, 0.0, 0.5075366, 0.5610923, 0.22793348, 0.15701589,
      0.21100126, 0.28684008, 0.48019138, 0.5790498, 0.0, 0.0, 0.51299816, 0.5537319, 0.37157783,
      0.42960122, 0.4258391, 0.25461295, 0.07401075, 0.36288196, 0.46194813, 0.36622074,
      0.40474024, 0.42051914, 0.30529416, 0.35716558, 0.4104683, 0.3634557, 0.18230169,
      0.12811545, 0.5179936, 0.559532, 0.3976659, 0.35664213, 0.19423184, 0.7187436, 0.73311293,
      0.66162705, 0.40453663, 0.40114495, 0.41675943, 0.39625713, 0.39887276, 0.48166054,
      0.50793016, 0.56699604, 0.3960097, 0.4437459, 0.69167227, 0.5400801, 0.652137,
      0.8049347, 0.8948945, 0.86430603, 0.600158, 0.50343186, 0.20795111, 0.23471259,
      0.39646378, 0.07110661, 0.20663853, 0.95886403, 0.73128873, 0.5521641, 0.08182517,
      0.4551149, 0.6015099, 0.36086103, 0.44722062, 0.100247495, 0.0, 0.0, 0.42081234, 0.34027046,
      0.4018491, 0.54632396, 0.86398995, 0.27266297, 0.48549172, 0.21826318, 0.0, 0.0, 0.0, 0.0,
      0.6506955, 0.27804166, 0.75620246, 0.69736207, 0.8209634, 0.0, 0.0), Array(10, 10))
    val dstTensor = Tensor[Double](
      storage = Storage[Double](dst[Tensor[Float]](ImageFeature.imageTensor).storage().array()
        .map(_.toDouble)), storageOffset = 1, size = Array(10, 10, 1))
    dstTensor.view(10, 10).map(dstTorch, (v1, v2) => {
      assert(math.abs(v1-v2)<1e-6)
      v1
    })

  }
}
