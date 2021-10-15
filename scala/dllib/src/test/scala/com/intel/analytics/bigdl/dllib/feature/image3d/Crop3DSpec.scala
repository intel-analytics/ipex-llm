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

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.dllib.utils.RandomGenerator._
import org.scalatest.{FlatSpec, Matchers}

class Crop3DSpec extends FlatSpec with Matchers{
  "A CropTransformer" should "generate correct output." in{
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](60, 70, 80)
    input.apply1(e => RNG.uniform(0, 1).toFloat)
    input.resize(60, 70, 80, 1)
    val image = ImageFeature3D(input)
    val start = Array[Int](10, 20, 20)
    val patchSize = Array[Int](21, 31, 41)
    val cropper = Crop3D(start, patchSize)
    val output = cropper.transform(image)
    val result = input.narrow(1, 10, 21).narrow(2, 20, 31).narrow(3, 20, 41)
      .storage().array()
    output[Tensor[Float]](ImageFeature.imageTensor).storage().array() should be(result)
  }

  "A RandomCropTransformer" should "generate correct output." in{
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](60, 70, 80)
    input.apply1(e => RNG.uniform(0, 1).toFloat)
    input.resize(60, 70, 80, 1)
    val image = ImageFeature3D(input)
    val cropper = RandomCrop3D(20, 30, 40)
    val output = cropper.transform(image)
    assert(output[Tensor[Float]](ImageFeature.imageTensor).size(1) == 20 )
    assert(output[Tensor[Float]](ImageFeature.imageTensor).size(2) == 30)
    assert(output[Tensor[Float]](ImageFeature.imageTensor).size(3) == 40)
  }
}
