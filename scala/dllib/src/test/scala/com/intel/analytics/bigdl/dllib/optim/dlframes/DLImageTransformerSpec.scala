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

package com.intel.analytics.bigdl.dlframes

import com.intel.analytics.bigdl.transform.vision.image.{ImageFrame, ImageFrameToSample, MatToTensor}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Row, SQLContext}
import org.opencv.core.CvType
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

import scala.util.Random

class DLImageTransformerSpec extends FlatSpec with Matchers with BeforeAndAfter {
  private var sc : SparkContext = _
  private var sqlContext : SQLContext = _
  private val pascalResource = getClass.getClassLoader.getResource("pascal/")
  private val imageNetResource = getClass.getClassLoader.getResource("imagenet/")

  before {
    val conf = Engine.createSparkConf().setAppName("Test DLImageTransfomer").setMaster("local[1]")
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

  "DLTransformer" should "setters work" in {
    val transformer = Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(123, 117, 104, 1, 1, 1) -> MatToTensor()
    val trans = new DLImageTransformer(transformer)
      .setInputCol("image1")
      .setOutputCol("features1")
    assert(trans.getInputCol == "image1")
    assert(trans.getOutputCol == "features1")
  }

  "DLTransformer" should "has correct result with pascal images" in {
    val imageDF = DLImageReader.readImages(pascalResource.getFile, sc)
    assert(imageDF.count() == 1)
    val transformer = Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(123, 117, 104, 1, 1, 1) -> MatToTensor()
    val transformedDF = new DLImageTransformer(transformer)
      .setInputCol("image")
      .setOutputCol("features")
      .transform(imageDF)
    val r = transformedDF.select("features").rdd.first().getAs[Row](0)
    assert(r.getString(0).endsWith("pascal/000025.jpg"))
    assert(r.getInt(1) == 224)
    assert(r.getInt(2) == 224)
    assert(r.getInt(3) == 3)
    assert(r.getInt(4) == CvType.CV_32FC3)
    assert(r.getSeq[Float](5).take(6).toArray.deep == Array(-30, -50, -69, -84, -46, -25).deep)
  }

  "DLTransformer" should "has correct result without MatToTensor" in {
    val imageDF = DLImageReader.readImages(pascalResource.getFile, sc)
    assert(imageDF.count() == 1)
    val transformer = Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(123, 117, 104, 1, 1, 1)
    val transformedDF = new DLImageTransformer(transformer)
      .setInputCol("image")
      .setOutputCol("features")
      .transform(imageDF)
    val r = transformedDF.select("features").rdd.first().getAs[Row](0)
    assert(r.getString(0).endsWith("pascal/000025.jpg"))
    assert(r.getInt(1) == 224)
    assert(r.getInt(2) == 224)
    assert(r.getInt(3) == 3)
    assert(r.getInt(4) == CvType.CV_32FC3)
    assert(r.getSeq[Float](5).take(6).toArray.deep == Array(-30, -50, -69, -84, -46, -25).deep)
  }

  "DLTransformer" should "ensure imf2Row and Row2Imf reversible" in {
    val imageDF = DLImageReader.readImages(pascalResource.getFile, sc)
    assert(imageDF.count() == 1)
    val transformer = Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(123, 117, 104, 1, 1, 1) -> MatToTensor()
    val transformedDF = new DLImageTransformer(transformer)
      .setInputCol("image")
      .setOutputCol("features")
      .transform(imageDF)
    val r = transformedDF.select("features").rdd.first().getAs[Row](0)
    val convertedR = DLImageSchema.imf2Row(DLImageSchema.row2IMF(r))

    assert(r.getSeq[Float](5).toArray.deep == convertedR.getAs[Array[Float]](5).deep)
  }

  "DLTransformer" should "transform gray scale image" in {
    val resource = getClass().getClassLoader().getResource("gray/gray.bmp")
    val df = DLImageReader.readImages(resource.getFile, sc)
    val dlTransformer = new DLImageTransformer(Resize(28, 28) -> MatToTensor[Float]())
      .setInputCol("image")
      .setOutputCol("features")
    val r = dlTransformer.transform(df).select("features").rdd.first().getAs[Row](0)
    assert(r.getString(0).endsWith("gray.bmp"))
    assert(r.getInt(1) == 28)
    assert(r.getInt(2) == 28)
    assert(r.getInt(3) == 1)
    assert(r.getInt(4) == CvType.CV_32FC1)
  }

  "DLTransformer" should "report error with same input and output columns" in {
    val resource = getClass().getClassLoader().getResource("gray/gray.bmp")
    val df = DLImageReader.readImages(resource.getFile, sc)
    val dlTransformer = new DLImageTransformer(Resize(28, 28) -> MatToTensor[Float]())
      .setInputCol("image")
      .setOutputCol("image")
    intercept[IllegalArgumentException] {
      val transformed = dlTransformer.transform(df)
    }
  }

}
