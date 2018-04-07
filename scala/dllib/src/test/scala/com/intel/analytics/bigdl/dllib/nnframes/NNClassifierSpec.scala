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

package com.intel.analytics.zoo.pipeline.nnframes

import com.intel.analytics.bigdl.models.inception.Inception_v1
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.Trigger
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.MatToTensor
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SQLContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class NNClassifierSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc : SparkContext = _
  var sqlContext : SQLContext = _
  var smallData: Seq[(Array[Double], Double)] = _
  val nRecords = 100
  val maxEpoch = 20

  before {
    val conf = Engine.createSparkConf().setAppName("Test DLEstimator").setMaster("local[1]")
    sc = SparkContext.getOrCreate(conf)
    sqlContext = new SQLContext(sc)
    Random.setSeed(42)
    RNG.setSeed(42)
    Engine.init
  }

  after{
    if (sc != null) {
      sc.stop()
    }
  }

  "NNClasifier" should "support image FEATURE types" in {
    val pascalResource = getClass.getClassLoader.getResource("pascal/")
    val imageDF = NNImageReader.readImages(pascalResource.getFile, sc)
    assert(imageDF.count() == 1)
    val transformer = Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(123, 117, 104, 1, 1, 1) -> MatToTensor()
    val transformedDF = new NNImageTransformer(transformer)
      .setInputCol("image")
      .setOutputCol("features")
      .transform(imageDF)
      .withColumn("label", lit(2.0f))
    val estimator = new NNClassifier(
      Inception_v1(1000), ClassNLLCriterion[Float](), Array(3, 224, 224))
      .setBatchSize(1)
      .setEndWhen(Trigger.maxIteration(1))
    estimator.fit(transformedDF)
  }
}
