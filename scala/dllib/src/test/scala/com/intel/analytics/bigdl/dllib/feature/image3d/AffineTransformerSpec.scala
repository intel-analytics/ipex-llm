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
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image._
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.utils.RandomGenerator._
import com.intel.analytics.bigdl.dllib.feature.image.ImageSet
import org.apache.spark.SparkContext
import org.scalatest.{FlatSpec, Matchers}

class AffineTransformerSpec extends FlatSpec with Matchers{
  "An AffineTransformer" should "generate correct output when dimension of depth is 1" in {
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
    val dstTorch = Tensor[Double](Array[Double](0.5434049386531115, 0.5434049386531115,
      0.5434049386531115, 0.5434049386531115, 0.5434049386531115, 0.5434049386531115,
      0.5434049386531115, 0.5434049386531115, 0.6030485902295856, 0.34584709625329957,
      0.5434049386531115, 0.5434049386531115, 0.5434049386531115, 0.5434049386531115,
      0.5434049386531115, 0.5434049386531115, 0.5792170105142169, 0.43088245638626355,
      0.404440362729813, 0.4985531174547446, 0.5434049386531115, 0.5434049386531115,
      0.5434049386531115, 0.5434049386531115, 0.5434049386531115, 0.6352126561832723,
      0.28505784836271975, 0.4362795138389034, 0.5169354642558416, 0.1866911782034402,
      0.5434049386531115, 0.5434049386531115, 0.5434049386531115, 0.5434049386531115,
      0.4363941840183775, 0.42806906843183085, 0.28031103085267134, 0.2991110983631679,
      0.48206986095900906, 0.1872089144539782, 0.5434049386531115, 0.4142729943412573,
      0.15258750343945277, 0.40309160921458287, 0.35514042215964303, 0.26149009155589875,
      0.5681985451405007, 0.2379225472454488, 0.15216648054234572, 0.4440014433445447,
      0.5386876560435572, 0.7690923666176646, 0.7718087438927854, 0.5285612710211486,
      0.744568241457456, 0.7328413136621909, 0.2421084535394531, 0.4398024047940161,
      0.4130078941327757, 0.08845560964899934, 0.8335512422334377, 0.6870108375950857,
      0.3453873219638647, 0.5181416388163627, 0.48346394942351695, 0.5249326909069478,
      0.1760288526635031, 0.06186541763270581, 0.048420564737170935, 0.048420564737170935,
      0.45021659374098083, 0.18498730051185502, 0.7016652106321166, 0.635941540821923,
      0.3230204399592828, 0.048420564737170935, 0.048420564737170935, 0.048420564737170935,
      0.048420564737170935, 0.048420564737170935, 0.34397570975731173, 0.056832111041893445,
      0.3606082114463558, 0.3104244235442204, 0.048420564737170935, 0.048420564737170935,
      0.048420564737170935, 0.048420564737170935, 0.048420564737170935, 0.048420564737170935,
      0.1624663604471237, 0.6682567507337012, 0.048420564737170935, 0.048420564737170935,
      0.048420564737170935, 0.048420564737170935, 0.048420564737170935, 0.048420564737170935,
      0.048420564737170935, 0.048420564737170935), Array(10, 10))
    val dstTensor = Tensor[Double](
      storage = Storage[Double](dst[Tensor[Float]](ImageFeature.imageTensor).storage().array()
        .map(_.toDouble)), storageOffset = 1, size = Array(1, 10, 10))
    dstTensor.view(10, 10).map(dstTorch, (v1, v2) => {
      assert(math.abs(v1-v2)<1e-6)
      v1
    })
  }

  "An AffineTransformer" should "generate correct output when dimension of height is 1" in {
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
    val dstTorch = Tensor[Double](Array[Double](0.5434049386531115, 0.5434049386531115,
      0.5434049386531115, 0.5434049386531115, 0.5434049386531115, 0.5434049386531115,
      0.5434049386531115, 0.5434049386531115, 0.6030485902295856, 0.34584709625329957,
      0.5434049386531115, 0.5434049386531115, 0.5434049386531115, 0.5434049386531115,
      0.5434049386531115, 0.5434049386531115, 0.5792170105142169, 0.43088245638626355,
      0.404440362729813, 0.4985531174547446, 0.5434049386531115, 0.5434049386531115,
      0.5434049386531115, 0.5434049386531115, 0.5434049386531115, 0.6352126561832723,
      0.28505784836271975, 0.4362795138389034, 0.5169354642558416, 0.1866911782034402,
      0.5434049386531115, 0.5434049386531115, 0.5434049386531115, 0.5434049386531115,
      0.4363941840183775, 0.42806906843183085, 0.28031103085267134, 0.2991110983631679,
      0.48206986095900906, 0.1872089144539782, 0.5434049386531115, 0.4142729943412573,
      0.15258750343945277, 0.40309160921458287, 0.35514042215964303, 0.26149009155589875,
      0.5681985451405007, 0.2379225472454488, 0.15216648054234572, 0.4440014433445447,
      0.5386876560435572, 0.7690923666176646, 0.7718087438927854, 0.5285612710211486,
      0.744568241457456, 0.7328413136621909, 0.2421084535394531, 0.4398024047940161,
      0.4130078941327757, 0.08845560964899934, 0.8335512422334377, 0.6870108375950857,
      0.3453873219638647, 0.5181416388163627, 0.48346394942351695, 0.5249326909069478,
      0.1760288526635031, 0.06186541763270581, 0.048420564737170935, 0.048420564737170935,
      0.45021659374098083, 0.18498730051185502, 0.7016652106321166, 0.635941540821923,
      0.3230204399592828, 0.048420564737170935, 0.048420564737170935, 0.048420564737170935,
      0.048420564737170935, 0.048420564737170935, 0.34397570975731173, 0.056832111041893445,
      0.3606082114463558, 0.3104244235442204, 0.048420564737170935, 0.048420564737170935,
      0.048420564737170935, 0.048420564737170935, 0.048420564737170935, 0.048420564737170935,
      0.1624663604471237, 0.6682567507337012, 0.048420564737170935, 0.048420564737170935,
      0.048420564737170935, 0.048420564737170935, 0.048420564737170935, 0.048420564737170935,
      0.048420564737170935, 0.048420564737170935), Array(10, 10))
    val dstTensor = Tensor[Double](
      storage = Storage[Double](dst[Tensor[Float]](ImageFeature.imageTensor).storage().array()
        .map(_.toDouble)), storageOffset = 1, size = Array(10, 1, 10))
    dstTensor.view(10, 10).map(dstTorch, (v1, v2) => {
      assert(math.abs(v1-v2)<1e-6)
      v1
    })
  }

  "An AffineTransformer" should "generate correct output when dimension of width is 1" in {
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
    val dstTorch = Tensor[Double](Array[Double](0.5434049386531115, 0.5434049386531115,
      0.5434049386531115, 0.5434049386531115, 0.5434049386531115, 0.5434049386531115,
      0.5434049386531115, 0.5434049386531115, 0.6030485902295856, 0.34584709625329957,
      0.5434049386531115, 0.5434049386531115, 0.5434049386531115, 0.5434049386531115,
      0.5434049386531115, 0.5434049386531115, 0.5792170105142169, 0.43088245638626355,
      0.404440362729813, 0.4985531174547446, 0.5434049386531115, 0.5434049386531115,
      0.5434049386531115, 0.5434049386531115, 0.5434049386531115, 0.6352126561832723,
      0.28505784836271975, 0.4362795138389034, 0.5169354642558416, 0.1866911782034402,
      0.5434049386531115, 0.5434049386531115, 0.5434049386531115, 0.5434049386531115,
      0.4363941840183775, 0.42806906843183085, 0.28031103085267134, 0.2991110983631679,
      0.48206986095900906, 0.1872089144539782, 0.5434049386531115, 0.4142729943412573,
      0.15258750343945277, 0.40309160921458287, 0.35514042215964303, 0.26149009155589875,
      0.5681985451405007, 0.2379225472454488, 0.15216648054234572, 0.4440014433445447,
      0.5386876560435572, 0.7690923666176646, 0.7718087438927854, 0.5285612710211486,
      0.744568241457456, 0.7328413136621909, 0.2421084535394531, 0.4398024047940161,
      0.4130078941327757, 0.08845560964899934, 0.8335512422334377, 0.6870108375950857,
      0.3453873219638647, 0.5181416388163627, 0.48346394942351695, 0.5249326909069478,
      0.1760288526635031, 0.06186541763270581, 0.048420564737170935, 0.048420564737170935,
      0.45021659374098083, 0.18498730051185502, 0.7016652106321166, 0.635941540821923,
      0.3230204399592828, 0.048420564737170935, 0.048420564737170935, 0.048420564737170935,
      0.048420564737170935, 0.048420564737170935, 0.34397570975731173, 0.056832111041893445,
      0.3606082114463558, 0.3104244235442204, 0.048420564737170935, 0.048420564737170935,
      0.048420564737170935, 0.048420564737170935, 0.048420564737170935, 0.048420564737170935,
      0.1624663604471237, 0.6682567507337012, 0.048420564737170935, 0.048420564737170935,
      0.048420564737170935, 0.048420564737170935, 0.048420564737170935, 0.048420564737170935,
      0.048420564737170935, 0.048420564737170935), Array(10, 10))
    val dstTensor = Tensor[Double](
      storage = Storage[Double](dst[Tensor[Float]](ImageFeature.imageTensor).storage().array()
        .map(_.toDouble)), storageOffset = 1, size = Array(10, 10, 1))
    dstTensor.view(10, 10).map(dstTorch, (v1, v2) => {
      assert(math.abs(v1-v2)<1e-6)
      v1
    })

  }
}
