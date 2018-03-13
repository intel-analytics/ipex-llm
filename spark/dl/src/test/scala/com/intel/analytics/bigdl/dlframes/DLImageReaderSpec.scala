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

import com.intel.analytics.bigdl.transform.vision.image.{ImageFrame, MatToTensor}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.Resize
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Row, SQLContext}
import org.opencv.core.CvType
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class DLImageReaderSpec extends FlatSpec with Matchers with BeforeAndAfter {

  var sc : SparkContext = _
  var sQLContext: SQLContext = _
  val pascalResource = getClass.getClassLoader.getResource("pascal/")
  private val imageNetResource = getClass.getClassLoader.getResource("imagenet/")

  before {
    val conf = Engine.createSparkConf().setAppName("Test DLImageReader").setMaster("local[1]")
    sc = SparkContext.getOrCreate(conf)
    sQLContext = new SQLContext(sc)

    Random.setSeed(42)
    RNG.setSeed(42)

    Engine.init
  }

  after{
    if (sc != null) {
      sc.stop()
    }
  }

  "DLImageReader" should "has correct result for pascal" in {
    val imageDF = DLImageReader.readImages(pascalResource.getFile, sc)
    assert(imageDF.count() == 1)
    val r = imageDF.head().getAs[Row](0)
    assert(r.getString(0).endsWith("000025.jpg"))
    assert(r.getInt(1) == 375)
    assert(r.getInt(2) == 500)
    assert(r.getInt(3) == 3)
    assert(r.getInt(4) == CvType.CV_8UC3)
    assert(r.getAs[Array[Byte]](5).length == 95959)
  }

  "DLImageReader" should "has correct result for imageNet" in {
    val imageDirectory = imageNetResource + "n02110063/"
    val imageDF = DLImageReader.readImages(imageDirectory, sc)
    assert(imageDF.count() == 3)
    val expectedRows = Seq(
      (imageDirectory + "n02110063_8651.JPEG", 99, 129, 3, CvType.CV_8UC3),
      (imageDirectory + "n02110063_11239.JPEG", 333, 500, 3, CvType.CV_8UC3),
      (imageDirectory + "n02110063_15462.JPEG", 332, 500, 3, CvType.CV_8UC3)
    )
    val actualRows = imageDF.rdd.collect().map(r => r.getAs[Row](0)).map { r =>
      (r.getString(0), r.getInt(1), r.getInt(2), r.getInt(3), r.getInt(4))
    }
    assert (expectedRows.toSet == actualRows.toSet)
  }

  "DLImageReader" should "has correct result for imageNet with channel 1 and 4" in {
    val imageDirectory = imageNetResource + "n99999999/"
    val imageDF = DLImageReader.readImages(imageDirectory, sc)
    assert(imageDF.count() == 3)
    val expectedRows = Seq(
      (imageDirectory + "n02105855_2933.JPEG", 189, 213, 4, CvType.CV_8UC4),
      (imageDirectory + "n02105855_test1.bmp", 527, 556, 1, CvType.CV_8UC1),
      (imageDirectory + "n03000134_4970.JPEG", 480, 640, 3, CvType.CV_8UC3)
    )
    val actualRows = imageDF.rdd.collect().map(r => r.getAs[Row](0)).map { r =>
      (r.getString(0), r.getInt(1), r.getInt(2), r.getInt(3), r.getInt(4))
    }
    assert (expectedRows.toSet == actualRows.toSet)
  }

  "DLImageReader" should "read recursively by wildcard path" in {
    val imageDF = DLImageReader.readImages(imageNetResource.getFile + "*", sc)
    assert(imageDF.count() == 11)
  }

  "DLImageReader" should "read from multiple path" in {
    val imageDirectory1 = imageNetResource + "n02110063/"
    val imageDirectory2 = imageNetResource + "n99999999/"
    val imageDF = DLImageReader.readImages(imageDirectory1 + "," + imageDirectory2, sc)
    assert(imageDF.count() == 6)
  }

  "read gray scale image" should "work" in {
    val resource = getClass().getClassLoader().getResource("gray/gray.bmp")
    val df = DLImageReader.readImages(resource.getFile, sc)
    assert(df.count() == 1)
    val r = df.head().getAs[Row](0)
    assert(r.getString(0).endsWith("gray.bmp"))
    assert(r.getInt(1) == 50)
    assert(r.getInt(2) == 50)
    assert(r.getInt(3) == 1)
    assert(r.getInt(4) == CvType.CV_8UC1)
  }
}
