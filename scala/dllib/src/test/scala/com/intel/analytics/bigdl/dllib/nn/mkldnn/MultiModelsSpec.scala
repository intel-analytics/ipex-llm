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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame, ImageFrameToSample, MatToTensor}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class MultiModelsSpec extends FlatSpec with BeforeAndAfter with Matchers {
  var sc: SparkContext = _
  val nodeNumber = 1
  val coreNumber = 2

  before {
    System.setProperty("bigdl.engineType", "mkldnn")
    System.setProperty("bigdl.multiModels", "true")

    LoggerFilter.redirectSparkInfoLogs()

    val conf = Engine.createSparkConf()
      .setMaster(s"local[$coreNumber]")
      .setAppName("multi mkl-dnn models")
    sc = SparkContext.getOrCreate(conf)

    Engine.init
  }

  after {
    System.clearProperty("bigdl.engineType")
    System.clearProperty("bigdl.multiModels")

    if (sc != null) {
      sc.stop()
    }
  }

  "model.predict" should "be correct" in {
    RNG.setSeed(100)
    val data = new Array[Sample[Float]](97)
    var i = 0
    while (i < data.length) {
      val input = Tensor[Float](28, 28).apply1(_ =>
        RNG.uniform(0.130660 + i, 0.3081078).toFloat)
      val label = Tensor[Float](1).fill(1.0f)
      data(i) = Sample(input, label)
      i += 1
    }
    val model = LeNet5(classNum = 10)
    val dataSet = sc.parallelize(data, coreNumber)

    var result = model.predict(dataSet)
    var prob = result.collect()

    prob(0) should be (model.forward(data(0).feature))
    prob(11) should be (model.forward(data(11).feature))
    prob(31) should be (model.forward(data(31).feature))
    prob(51) should be (model.forward(data(51).feature))
    prob(71) should be (model.forward(data(71).feature))
    prob(91) should be (model.forward(data(91).feature))

    result = model.predict(dataSet, 20, true)
    prob = result.collect()

    prob(0) should be(prob(10))
    prob(5) should be(prob(15))
    prob(0) should be(prob(20))
    prob(8) should be(prob(38))
  }

  "model.predictClass" should "be correct" in {
    RNG.setSeed(100)
    val data = new Array[Sample[Float]](97)
    var i = 0
    while (i < data.length) {
      val input = Tensor[Float](28, 28).apply1(_ =>
        RNG.uniform(0.130660 + i, 0.3081078).toFloat)
      val label = Tensor[Float](1).fill(1.0f)
      data(i) = Sample(input, label)
      i += 1
    }
    val model = LeNet5(classNum = 10)
    val dataSet = sc.parallelize(data, 2)
    val result = model.predictClass(dataSet)

    val prob = result.collect()
    prob(0) should be
    (model.forward(data(0).feature
    ).toTensor[Float].max(1)._2.valueAt(1).toInt)
    prob(11) should be
    (model.forward(data(11).feature
    ).toTensor[Float].max(1)._2.valueAt(1).toInt)
    prob(31) should be
    (model.forward(data(31).feature
    ).toTensor[Float].max(1)._2.valueAt(1).toInt)
    prob(51) should be
    (model.forward(data(51).feature
    ).toTensor[Float].max(1)._2.valueAt(1).toInt)
    prob(71) should be
    (model.forward(data(71).feature
    ).toTensor[Float].max(1)._2.valueAt(1).toInt)
    prob(91) should be
    (model.forward(data(91).feature
    ).toTensor[Float].max(1)._2.valueAt(1).toInt)
  }

  "model.predictImage" should "be correct" in {
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    RNG.setSeed(100)
    val resource = getClass.getClassLoader.getResource("pascal/")
    val imageFrame = ImageFrame.read(resource.getFile, sc) ->
      Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      MatToTensor() -> ImageFrameToSample()
    val model = Inception_v1_NoAuxClassifier(classNum = 20)
    val detection = model.predictImage(imageFrame).toDistributed()
    val feature = detection.rdd.first()
    println(feature(ImageFeature.predict))

    val imageFeatures = detection.rdd.collect()
    val prob = imageFeatures.map(x => x[Tensor[Float]](ImageFeature.predict))
    val data = imageFeatures.map(_[Sample[Float]](ImageFeature.sample))
    prob(0) should be (model.forward(data(0).feature.reshape(Array(1, 3, 224, 224)))
      .toTensor[Float].squeeze)
  }
}
