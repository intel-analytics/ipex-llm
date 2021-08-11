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
package com.intel.analytics.bigdl.utils.intermediate

import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame, ImageFrameToSample, MatToTensor}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class DnnPredictorSpec extends FlatSpec with Matchers with BeforeAndAfter {
  private val coreNumber = 4
  private val nodeNumber = 1
  private var sc: SparkContext = _

  before {
    System.setProperty("bigdl.engineType", "mkldnn")
    Engine.init(nodeNumber, coreNumber, true)
    val conf = Engine.createSparkConf().setMaster(s"local[$coreNumber]").setAppName("dnn predictor")
    sc = SparkContext.getOrCreate(conf)
    Engine.init
  }

  after {
    if (sc != null) sc.stop()
    System.clearProperty("bigdl.engineType")
  }

  "predict image for dnn" should "work properly" in {
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    RNG.setSeed(100)
    val resource = getClass.getClassLoader.getResource("pascal/")
    val imageRead = ImageFrame.read(resource.getFile, sc) ->
      Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      MatToTensor() -> ImageFrameToSample()
    val imageFrame = ImageFrame.rdd(imageRead.toDistributed().rdd.repartition(coreNumber))
    val inception = Inception_v1_NoAuxClassifier(classNum = 20, false)
    val model = inception.cloneModule()
    val detection = model.predictImage(imageFrame).toDistributed()
    val feature = detection.rdd.first()
    println(feature(ImageFeature.predict))

    imageFrame.rdd.partitions.length should be(coreNumber)
    detection.rdd.partitions.length should be(1)

    val imageFeatures = detection.rdd.collect()
    val prob = imageFeatures.map(x => x[Tensor[Float]](ImageFeature.predict))
    val data = imageFeatures.map(_[Sample[Float]](ImageFeature.sample))

    prob(0) should be (model.forward(data(0).feature.reshape(Array(1, 3, 224, 224)))
      .toTensor[Float].squeeze)
  }

  "predict for dnn" should "work properly" in {
    RNG.setSeed(100)
    val data = new Array[Sample[Float]](97)
    var i = 0
    while (i < data.length) {
      val input = Tensor[Float](1, 28, 28).apply1(_ =>
        RNG.uniform(0.130660 + i, 0.3081078).toFloat)
      val label = Tensor[Float](1).fill(1.0f)
      data(i) = Sample(input, label)
      i += 1
    }
    val lenet = LeNet5(classNum = 10)
    val model = lenet.cloneModule()
    val dataSet = sc.parallelize(data, coreNumber).repartition(1)
    var result = model.predict(dataSet)
    var prob = result.collect()

    dataSet.partitions.length should be(1)
    result.partitions.length should be(1)

    prob(0) should be (model.forward(data(0).feature).toTensor[Float].squeeze())
    prob(11) should be (model.forward(data(11).feature).toTensor[Float].squeeze())
    prob(31) should be (model.forward(data(31).feature).toTensor[Float].squeeze())
    prob(51) should be (model.forward(data(51).feature).toTensor[Float].squeeze())
    prob(71) should be (model.forward(data(71).feature).toTensor[Float].squeeze())
    prob(91) should be (model.forward(data(91).feature).toTensor[Float].squeeze())

    val resultClass = model.predictClass(dataSet)

    val probClass = resultClass.collect()
    probClass(0) should be
    (model.forward(data(0).feature
    ).toTensor[Float].squeeze().max(1)._2.valueAt(1).toInt)
    probClass(11) should be
    (model.forward(data(11).feature
    ).toTensor[Float].squeeze().max(1)._2.valueAt(1).toInt)
    probClass(31) should be
    (model.forward(data(31).feature
    ).toTensor[Float].squeeze().max(1)._2.valueAt(1).toInt)
    probClass(51) should be
    (model.forward(data(51).feature
    ).toTensor[Float].squeeze().max(1)._2.valueAt(1).toInt)
    probClass(71) should be
    (model.forward(data(71).feature
    ).toTensor[Float].squeeze().max(1)._2.valueAt(1).toInt)
    probClass(91) should be
    (model.forward(data(91).feature
    ).toTensor[Float].squeeze().max(1)._2.valueAt(1).toInt)
  }

  "Evaluator with dnn backend" should "be correct" in {
    RNG.setSeed(100)
    val tmp = new Array[Sample[Float]](100)
    var i = 0
    while (i < tmp.length) {
      val input = Tensor[Float](28, 28).fill(0.8f)
      val label = Tensor[Float](1).fill(1.0f)
      tmp(i) = Sample(input, label)
      i += 1
    }
    val model = LeNet5(classNum = 10)
    val dataSet = DataSet.array(tmp, sc).toDistributed().data(train = false).repartition(4)

    val result = model.evaluate(dataSet, Array(new Top1Accuracy[Float](), new Top5Accuracy[Float](),
      new Loss[Float](CrossEntropyCriterion[Float]())))

    dataSet.partitions.length should be(4)
    result(0)._1 should be (new AccuracyResult(0, 100))
    result(1)._1 should be (new AccuracyResult(100, 100))
    result(2)._1 should be (new LossResult(230.44278f, 100))
    result(0)._1.result()._1 should be (0f)
    result(1)._1.result()._1 should be (1f)
    result(2)._1.result()._1 should be (2.3044279f+-0.000001f)
  }

  "Evaluator MiniBatch with dnn backend" should "be correct" in {
    RNG.setSeed(100)
    val tmp = new Array[MiniBatch[Float]](25)
    var i = 0
    while (i < tmp.length) {
      val input = Tensor[Float](4, 28, 28).fill(0.8f)
      val label = Tensor[Float](4).fill(1.0f)
      tmp(i) = MiniBatch(input, label)
      i += 1
    }
    val model = LeNet5(classNum = 10)
    val dataSet = DataSet.array(tmp, sc).toDistributed().data(train = false).repartition(4)

    val result = model.evaluate(dataSet, Array(new Top1Accuracy[Float](), new Top5Accuracy[Float](),
      new Loss[Float](CrossEntropyCriterion[Float]())))

    dataSet.partitions.length should be(4)
    result(0)._1 should be (new AccuracyResult(0, 100))
    result(1)._1 should be (new AccuracyResult(100, 100))
    result(2)._1 should be (new LossResult(230.44278f, 100))
    result(0)._1.result()._1 should be (0f)
    result(1)._1.result()._1 should be (1f)
    result(2)._1.result()._1 should be (2.3044279f+-0.000001f)
  }

  "Local predictor with dnn backend" should "work properly" in {
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    RNG.setSeed(100)
    val resource = getClass.getClassLoader.getResource("pascal/")
    val ims = (1 to 50).map(x => {
      val im = ImageFeature()
      im(ImageFeature.uri) = x.toString
      im(ImageFeature.imageTensor) = Tensor[Float](3, 24, 24).randn()
      im
    })

    val imageFrame = ImageFrame.array(ims.toArray) -> ImageFrameToSample()
    val m = Sequential()
    m.add(SpatialConvolution(3, 6, 5, 5))
    m.add(Tanh())
    val model = m.toGraph().asInstanceOf[StaticGraph[Float]]
    model.setInputFormats(Seq(Memory.Format.nchw))
    model.setOutputFormats(Seq(Memory.Format.nchw))

    val detection = model.predictImage(imageFrame).toLocal()
    val feature = detection.array.head

    val imageFeatures = detection.array
    val prob = imageFeatures.map(x => x[Tensor[Float]](ImageFeature.predict))
    val data = imageFeatures.map(_[Sample[Float]](ImageFeature.sample))
    (1 to 20).foreach(x => {
      imageFeatures(x - 1).uri() should be (x.toString)
      if (imageFeatures(x - 1).predict() == null) println(x, imageFeatures(x - 1).predict())
      assert(imageFeatures(x - 1).predict() != null)
    })
  }
}
