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

package com.intel.analytics.bigdl.optim

import java.io.File

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.{PaddingParam, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.quantized.StorageManager
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.commons.io.FileUtils
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class LocalPredictorSpec extends FlatSpec with Matchers with BeforeAndAfter {

  private val nodeNumber = 1
  private val coreNumber = 4
  val batchPerCore = 4
  var subModelNumber = coreNumber

  before {
    System.setProperty("bigdl.localMode", "true")
    Engine.init(nodeNumber, coreNumber, false)
    subModelNumber = Engine.getEngineType match {
      case MklBlas => coreNumber
      case MklDnn => 1
    }

  }

  after {
    System.clearProperty("bigdl.localMode")
  }

  "predictImage" should "work properly" in {
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    RNG.setSeed(100)
    val resource = getClass.getClassLoader.getResource("pascal/")
    val imageFrame = ImageFrame.read(resource.getFile) ->
      Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      MatToTensor() -> ImageFrameToSample()
    val model = Inception_v1_NoAuxClassifier(classNum = 20)
    val detection = model.predictImage(imageFrame).toLocal()
    val feature = detection.array.head
    println(feature(ImageFeature.predict))

    val imageFeatures = detection.array
    val prob = imageFeatures.map(x => x[Tensor[Float]](ImageFeature.predict))
    val data = imageFeatures.map(_[Sample[Float]](ImageFeature.sample))
    prob(0) should be(model.evaluate().forward(data(0).feature.reshape(Array(1, 3, 224, 224)))
      .toTensor[Float].squeeze())
  }

  "predictImage with more data" should "work properly" in {
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
    val model = Sequential()
    model.add(SpatialConvolution(3, 6, 5, 5))
    model.add(Tanh())
    val detection = model.predictImage(imageFrame).toLocal()
    val feature = detection.array.head

    val imageFeatures = detection.array
    val prob = imageFeatures.map(x => x[Tensor[Float]](ImageFeature.predict))
    val data = imageFeatures.map(_[Sample[Float]](ImageFeature.sample))
    prob(0) should be(model.evaluate().forward(data(0).feature.reshape(Array(1, 3, 24, 24)))
      .toTensor[Float].split(1)(0))
    (1 to 20).foreach(x => {
      imageFeatures(x - 1).uri() should be (x.toString)
      if (imageFeatures(x - 1).predict() == null) println(x, imageFeatures(x - 1).predict())
      assert(imageFeatures(x - 1).predict() != null)
    })
  }

  "predictImage with variant feature data" should "work" in {
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    RNG.setSeed(100)
    val ims = (1 to 50).map(x => {
      val size = RNG.uniform(20, 30).toInt
      val im = ImageFeature()
      im(ImageFeature.uri) = x.toString
      im(ImageFeature.imageTensor) = Tensor[Float](3, size, size).randn()
      im
    })

    val imageFrame = ImageFrame.array(ims.toArray) -> ImageFrameToSample()
    val model = Sequential()
    model.add(SpatialConvolution(3, 6, 5, 5))
    model.add(Tanh())
    val detection = model.predictImage(imageFrame, batchPerPartition = 1, featurePaddingParam =
      Some(PaddingParam())).toLocal()
    val imageFeatures = detection.array
    (1 to 20).foreach(x => {
      imageFeatures(x - 1).uri() should be (x.toString)
      println(imageFeatures(x - 1)[Tensor[Float]](ImageFeature.imageTensor).size().mkString("x"))
      println(imageFeatures(x - 1)[Sample[Float]](ImageFeature.sample)
        .getFeatureSize()(0).mkString("x"))
      println(x, imageFeatures(x - 1).predict().asInstanceOf[Tensor[Float]].size().mkString("x"))
      assert(imageFeatures(x - 1).predict() != null)
    })
  }

  "predictImage with quantize" should "work properly" in {
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    RNG.setSeed(100)
    val resource = getClass.getClassLoader.getResource("pascal/")
    val imageFrame = ImageFrame.read(resource.getFile) ->
      Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      MatToTensor() -> ImageFrameToSample()
    val model = Inception_v1_NoAuxClassifier(classNum = 20).quantize()
    val detection = model.predictImage(imageFrame).toLocal()
    val feature = detection.array.head
    println(feature(ImageFeature.predict))

    val imageFeatures = detection.array
    val prob = imageFeatures.map(x => x[Tensor[Float]](ImageFeature.predict))
    val data = imageFeatures.map(_[Sample[Float]](ImageFeature.sample))
    prob(0) should be(model.evaluate().forward(data(0).feature.reshape(Array(1, 3, 224, 224)))
      .toTensor[Float].squeeze())
  }

  "predictImage empty" should "work properly" in {
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    RNG.setSeed(100)
    val images = ImageFrame.array(Array[ImageFeature]())
    val imageFrame = images ->
      Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      MatToTensor() -> ImageFrameToSample()
    val model = Inception_v1_NoAuxClassifier(classNum = 20)
    val detection = model.predictImage(imageFrame).toLocal()

    val imageFeatures = detection.array
    imageFeatures.length should be (0)
  }

  "predictImage performance one by one" should "work properly" in {
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    RNG.setSeed(100)
    val resource = getClass.getClassLoader.getResource("pascal/")
    val imageFrame = ImageFrame.read(resource.getFile) ->
      Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      MatToTensor() -> ImageFrameToSample()
    val model = Inception_v1_NoAuxClassifier(classNum = 20)
    val localPredictor = LocalPredictor(model)

    model.forward(Tensor[Float](1, 3, 224, 224))

    var start = System.nanoTime()
    (1 to 20).foreach(x => {
      val detection = model.forward(Tensor[Float](1, 3, 224, 224))
    })
    println(s"${(System.nanoTime() - start) / 1e9}s")

    start = System.nanoTime()
    (1 to 20).foreach(x => {
      val detection = localPredictor.predictImage(imageFrame.toLocal()).toLocal()
    })

    println(s"${(System.nanoTime() - start) / 1e9}s")
  }

  "predictImage performance group" should "work properly" in {
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    RNG.setSeed(100)
    val resource = getClass.getClassLoader.getResource("pascal/000025.jpg")
    val imageFeatures = (1 to 20).map(i => {
      val f = new File(resource.getFile)
      ImageFeature(FileUtils.readFileToByteArray(f), f.getAbsolutePath)
    }).toArray
    val imageFrame = ImageFrame.array(imageFeatures) -> BytesToMat() ->
      Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      MatToTensor() -> ImageFrameToSample()
    val model = Inception_v1_NoAuxClassifier(classNum = 20)
    val localPredictor = LocalPredictor(model)

    model.forward(Tensor[Float](1, 3, 224, 224))
    var start = System.nanoTime()
    (1 to 20).foreach(x => {
      val detection = model.forward(Tensor[Float](1, 3, 224, 224))
    })
    println(s"${(System.nanoTime() - start) / 1e9}s")

    start = System.nanoTime()
    val detection = localPredictor.predictImage(imageFrame.toLocal()).toLocal()

    println(s"${(System.nanoTime() - start) / 1e9}s")
  }

  "predict sample after refactor" should "work properly" in {
    val samples = (1 to 20).map(i => {
      Sample(Tensor[Float](3, 224, 224).randn())
    }).toArray
    val imageFrame = ImageFrame.array((0 until 20).map(x => {
      val im = ImageFeature()
      im(ImageFeature.sample) = samples(x)
      im
    }).toArray)

    val model = Inception_v1_NoAuxClassifier(classNum = 20)
    val out1 = model.predictImage(imageFrame).toLocal().array
      .map(_.predict().asInstanceOf[Tensor[Float]])
    val out2 = predict(samples, model)

    out1.zip(out2).foreach(x => {
      x._1 should be (x._2.toTensor[Float])
    })

  }

  def predict(dataSet: Array[Sample[Float]], model: Module[Float]): Array[Activity] = {
    val weightsBias = Util.getAndClearWeightBias[Float](model.cloneModule().parameters())
    val iter = dataSet.iterator
    val transformer = SampleToMiniBatch[Float](
      batchSize = batchPerCore * subModelNumber, None, None,
      partitionNum = Some(1))
    val dataIter = transformer(iter)

    dataIter.map(batch => {
      val stackSize = batch.size() / subModelNumber
      val extraSize = batch.size() % subModelNumber
      val parallelism = if (stackSize == 0) extraSize else subModelNumber
      val workingModels = (1 to subModelNumber).map(_ => {
        val submodel = model.cloneModule().evaluate()
        Util.putWeightBias(weightsBias, submodel)
        submodel
      }).toArray
      val start = System.nanoTime()
      val result = Engine.default.invokeAndWait(
        (0 until parallelism).map(b =>
          () => {
            val offset = b * stackSize + math.min(b, extraSize) + 1
            val length = stackSize + (if (b < extraSize) 1 else 0)
            val currentMiniBatch = batch.slice(offset, length)
            val input = currentMiniBatch.getInput()
            val output = workingModels(b).forward(input).toTensor[Float]
            output.clone()
          }
        )
      )
      val batchResult = result.flatMap(_.split(1)).map(_.asInstanceOf[Activity])
      batchResult
    }).toArray.flatten

  }

  "predictImage with table output" should "work properly" in {
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
    val input = Input()
    val conv = SpatialConvolution(3, 6, 5, 5).inputs(input)
    val out1 = Tanh().inputs(conv)
    val out2 = ReLU().inputs(conv)
    val model = Graph(input, Array(out1, out2))
    val detection = model.predictImage(imageFrame).toLocal()
    val feature = detection.array.head

    val imageFeatures = detection.array
    (1 to 20).foreach(x => {
      imageFeatures(x - 1).uri() should be (x.toString)
      assert(imageFeatures(x - 1).predict() != null)
      assert(imageFeatures(x - 1).predict().asInstanceOf[Table].length() == 2)
    })
  }
}
