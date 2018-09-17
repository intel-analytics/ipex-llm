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

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.zoo.feature.{TH, TorchSpec}
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.feature.image.ImageSet
import org.apache.spark.SparkContext

class AffineTransformerSpec extends TorchSpec {
  "An AffineTransformer" should "generate correct output when dimension of depth is 1" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](1, 10, 10)
    input.apply1(e => RNG.uniform(0, 1))

    val a = RNG.uniform(0, 2)
    val b = RNG.uniform(0, 2)
    val c = RNG.uniform(0, 2)
    val d = RNG.uniform(0, 2)
    val e = RNG.uniform(0, 2)
    val f = RNG.uniform(0, 2)

    val matArray = Array[Double](1, 0, 0, 0, a, b, 0, c, d)
    val matTensor = Tensor[Double](matArray, Array[Int](3, 3))
    val translation = Tensor[Double](3)
    translation(1) = 0
    translation(2) = e
    translation(3) = f
    val translation2 = Tensor[Double](2)
    translation2(1) = e
    translation2(2) = f
    val mat2Array = Array[Double](a, b, c, d)
    val mat2Tensor = Tensor[Double](mat2Array, Array[Int](2, 2))
    val aff = AffineTransform3D(matTensor, translation = translation)
    val dims = Array[Int](1, 10, 10)
    val tensor = Tensor[Float](
      storage = Storage[Float](input.storage().array().map(_.toFloat)),
      storageOffset = 1,
      size = Array(1, 10, 10, 1))
    val image = ImageFeature3D(tensor)
    val conf = Engine.createSparkConf().setAppName("Test NNClassifier").setMaster("local[1]")
    val sc = SparkContext.getOrCreate(conf)
    val rdd = sc.parallelize(Seq[ImageFeature](image))
    val imageSet = ImageSet.rdd(rdd)
    val dst = aff.transform(image)
    val code = "require 'image'\n" +
    "dst = image.affinetransform(src, mat, 'bilinear', translation)"
    val (luaTime, torchResult) = TH.run(code,
      Map("src" -> input.view(10, 10), "mat" -> mat2Tensor, "translation" -> translation2),
      Array("dst"))
    val dstTorch = torchResult("dst").asInstanceOf[Tensor[Double]]
    val dstTensor = Tensor[Double](
      storage = Storage[Double](dst[Tensor[Float]](ImageFeature.imageTensor).storage().array()
        .map(_.toDouble)), storageOffset = 1, size = Array(1, 10, 10))
    dstTensor.view(10, 10).map(dstTorch, (v1, v2) => {
      assert(math.abs(v1-v2)<1e-6)
      v1
    })
  }

  "An AffineTransformer" should "generate correct output when dimension of height is 1" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](10, 1, 10)
    input.apply1(e => RNG.uniform(0, 1))

    val a = RNG.uniform(0, 2)
    val b = RNG.uniform(0, 2)
    val c = RNG.uniform(0, 2)
    val d = RNG.uniform(0, 2)
    val e = RNG.uniform(0, 2)
    val f = RNG.uniform(0, 2)

    val matArray = Array[Double](a, 0, b, 0, 1, 0, c, 0, d)
    val matTensor = Tensor[Double](matArray, Array[Int](3, 3))
    val translation = Tensor[Double](3)
    translation(1) = e
    translation(2) = 0
    translation(3) = f
    val translation2 = Tensor[Double](2)
    translation2(1) = e
    translation2(2) = f
    val mat2Array = Array[Double](a, b, c, d)
    val mat2Tensor = Tensor[Double](mat2Array, Array[Int](2, 2))
    val aff = AffineTransform3D(matTensor, translation = translation)
    val tensor = Tensor[Float](
      storage = Storage[Float](input.storage().array().map(_.toFloat)),
      storageOffset = 1,
      size = Array(10, 1, 10, 1))
    val image = ImageFeature3D(tensor)
    val dst = aff.transform(image)
    val code = "require 'image'\n" +
    "dst = image.affinetransform(src,mat,'bilinear',translation)"
    val (luaTime, torchResult) = TH.run(code,
      Map("src" -> input.view(10, 10), "mat" -> mat2Tensor, "translation" -> translation2),
      Array("dst"))
    val dstTorch = torchResult("dst").asInstanceOf[Tensor[Double]]
    val dstTensor = Tensor[Double](
      storage = Storage[Double](dst[Tensor[Float]](ImageFeature.imageTensor).storage().array()
        .map(_.toDouble)), storageOffset = 1, size = Array(10, 1, 10))
    dstTensor.view(10, 10).map(dstTorch, (v1, v2) => {
      assert(math.abs(v1-v2)<1e-6)
      v1
    })
  }

  "An AffineTransformer" should "generate correct output when dimension of width is 1" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](10, 10, 1)
    input.apply1(e => RNG.uniform(0, 1))

    val a = RNG.uniform(0, 2)
    val b = RNG.uniform(0, 2)
    val c = RNG.uniform(0, 2)
    val d = RNG.uniform(0, 2)
    val e = RNG.uniform(0, 2)
    val f = RNG.uniform(0, 2)

    val matArray = Array[Double](a, b, 0, c, d, 0, 0, 0, 1)
    val matTensor = Tensor[Double](matArray, Array[Int](3, 3))
    val translation = Tensor[Double](3)
    translation(1) = e
    translation(2) = f
    translation(3) = 0
    val translation2 = Tensor[Double](2)
    translation2(1) = e
    translation2(2) = f
    val mat2Array = Array[Double](a, b, c, d)
    val mat2Tensor = Tensor[Double](mat2Array, Array[Int](2, 2))
    val aff = AffineTransform3D(matTensor, translation = translation)
    val tensor = Tensor[Float](
      storage = Storage[Float](input.storage().array().map(_.toFloat)),
      storageOffset = 1,
      size = Array(10, 10, 1, 1))
    val image = ImageFeature3D(tensor)
    val dst = aff.transform(image)
    val code = "require 'image'\n" +
    "dst = image.affinetransform(src,mat,'bilinear',translation)"
    val (luaTime, torchResult) = TH.run(code,
      Map("src" -> input.view(10, 10), "mat" -> mat2Tensor, "translation" -> translation2),
      Array("dst"))
    val dstTorch = torchResult("dst").asInstanceOf[Tensor[Double]]
    val dstTensor = Tensor[Double](
      storage = Storage[Double](dst[Tensor[Float]](ImageFeature.imageTensor).storage().array()
        .map(_.toDouble)), storageOffset = 1, size = Array(10, 10, 1))
    dstTensor.view(10, 10).map(dstTorch, (v1, v2) => {
      assert(math.abs(v1-v2)<1e-6)
      v1
    })

  }
}
