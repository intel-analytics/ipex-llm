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

// NOTE: This spec is tested for native memory used. because we use singleton object to check
// how many pointer allocated, so all them should be in one file and serial executed.

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.Criterion
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.models.utils.CachedModels
import com.intel.analytics.bigdl.nn.CrossEntropyCriterion
import com.intel.analytics.bigdl.nn.quantized.{StorageInfo, StorageManager}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{DnnStorage, Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame, ImageFrameToSample, MatToTensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, RandomGenerator, SparkContextLifeCycle}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class OptimPredictorShutdownSpec extends SparkContextLifeCycle with Matchers {
  override def nodeNumber: Int = 1
  override def coreNumber: Int = 1
  override def appName: String = "predictor"

  "model predict should have no memory leak" should "be correct" in {
    LoggerFilter.redirectSparkInfoLogs()
    Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
    RNG.setSeed(100)
    val resource = getClass.getClassLoader.getResource("pascal/")
    val imageFrame = ImageFrame.read(resource.getFile, sc) ->
      Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      MatToTensor() -> ImageFrameToSample()
    val model = Inception_v1_NoAuxClassifier(classNum = 1000)
    val quant = model.quantize()
    val init = StorageManager.get()
    println(s"init count ${init.count(!_._2.isFreed)}")
    var second: Map[Long, StorageInfo] = null
    (0 until 20).foreach { i =>
      val detection = quant.predictImage(imageFrame, batchPerPartition = 16).toDistributed()
      detection.rdd.first()
      detection.rdd.collect()
      println("=" * 80)
      println(StorageManager.get().count(!_._2.isFreed))
      println("-" * 80)
    }
    CachedModels.deleteAll("")
    // NOTE: if this case failed, please check,
    // 1. mapPartition, does it used the variable out side of the method scope.
    // 2. ModelBroadcast, does it add the ref correctly
    StorageManager.get().count(!_._2.isFreed) should be (init.count(!_._2.isFreed))
  }

  "local predictor shutdown" should "work properly" in {
    val input = Tensor[Float](4, 3, 224, 224).rand(-1, 1)

    val samples = (1 to 20).map(i => {
      Sample(Tensor[Float](3, 224, 224).randn())
    }).toArray
    val imageFrame = ImageFrame.array((0 until 20).map(x => {
      val im = ImageFeature()
      im(ImageFeature.sample) = samples(x)
      im
    }).toArray)

    val model = Inception_v1_NoAuxClassifier(1000)
    val quant = model.quantize().evaluate()
    val initNativeSize = StorageManager.get().count(x => !x._2.isFreed)

    // has no memory issues
    (0 to 4).foreach { _ =>
      quant.predictImage(imageFrame).toLocal().array.map(_.predict().asInstanceOf[Tensor[Float]])
      StorageManager.get().count(x => !x._2.isFreed) should be (initNativeSize)
    }

    // check the model can work again
    quant.forward(input)
    val quant2 = model.quantize().evaluate()
    quant2.forward(input)

    quant.output.toTensor[Float] should be (quant2.output.toTensor[Float])
  }
}

class DistriOptimizerSpec2 extends SparkContextLifeCycle with Matchers {

  import DistriOptimizerSpecModel._

  override def appName: String = "RDDOptimizerSpec"

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  private var dataSet: DistributedDataSet[MiniBatch[Float]] = _

  override def beforeTest: Any = {
    System.setProperty("bigdl.engineType", "mkldnn")

    val input1: Tensor[Float] = Tensor[Float](Storage[Float](Array(0.0f, 1.0f, 0.0f, 1.0f)))
    val output1 = 0.0f
    val input2: Tensor[Float] = Tensor[Float](Storage[Float](Array(1.0f, 0.0f, 1.0f, 0.0f)))
    val output2 = 1.0f
    var plusOne = 1.0f
    val nodeNumber = 4
    val coreNumber = 4
    val batchSize = 2 * coreNumber
    Engine.init(nodeNumber, coreNumber, onSpark = true)

    val prepareData: Int => (MiniBatch[Float]) = index => {
      val input = Tensor[Float]().resize(batchSize, 4)
      val target = Tensor[Float]().resize(batchSize)
      var i = 0
      while (i < batchSize) {
        if (i % 2 == 0) {
          target.setValue(i + 1, output1 + plusOne)
          input.select(1, i + 1).copy(input1)
        } else {
          target.setValue(i + 1, output2 + plusOne)
          input.select(1, i + 1).copy(input2)
        }
        i += 1
      }
      MiniBatch(input, target)
    }

    val rdd = sc.parallelize(1 to (256 * 4), 4).map(prepareData)

    dataSet = new DistributedDataSet[MiniBatch[Float]] {
      override def originRDD(): RDD[_] = rdd

      override def data(train: Boolean): RDD[MiniBatch[Float]] = rdd

      override def size(): Long = rdd.count()

      override def shuffle(): Unit = {}
    }

    System.setProperty("bigdl.check.singleton", false.toString)
    Engine.model.setPoolSize(1)
  }

  override def afterTest: Any = {
    System.clearProperty("bigdl.engineType")
  }

  "Train model and shutdown" should "be good" in {
    RandomGenerator.RNG.setSeed(10)
    val model = dnn
    val count = DnnStorage.get().count(!_._2)
    val optimizer = new DistriOptimizer(
      model,
      dataSet,
      new CrossEntropyCriterion[Float]()).setEndWhen(Trigger.severalIteration(1))
    optimizer.optimize()
    DnnStorage.get().count(!_._2) should be (count)
  }
}

class LocalOptimizerSpec2 extends FlatSpec with Matchers with BeforeAndAfter {

  import DummyDataSet._
  import LocalOptimizerSpecModel._

  before {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.engineType", "mkldnn")
    Engine.init
  }

  after {
    System.clearProperty("bigdl.localMode")
    System.clearProperty("bigdl.engineType")
  }

  "Train model and shutdown" should "be good" in {
    RandomGenerator.RNG.setSeed(1000)
    val model = dnnModel
    val count = DnnStorage.get().count(!_._2)
    val optimizer = new LocalOptimizer[Float](
      model,
      creDataSet,
      new CrossEntropyCriterion[Float].asInstanceOf[Criterion[Float]]
    ).setEndWhen(Trigger.severalIteration(1))

    optimizer.optimize()
    DnnStorage.get().count(!_._2) should be (count)
  }
}
