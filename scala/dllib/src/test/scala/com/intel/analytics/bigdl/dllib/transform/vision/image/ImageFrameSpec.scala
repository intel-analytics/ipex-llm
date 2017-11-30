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

package com.intel.analytics.bigdl.transform.vision.image

import com.google.common.io.Files
import com.intel.analytics.bigdl.utils.Engine
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SQLContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ImageFrameSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val resource = getClass.getClassLoader.getResource("pascal/")
  var sc: SparkContext = null
  var sqlContext: SQLContext = null
  before {
    val conf = Engine.createSparkConf().setAppName("ImageSpec").setMaster("local[2]")
    sc = new SparkContext(conf)
    sqlContext = new SQLContext(sc)
  }

  after {
    if (null != sc) sc.stop()
  }

  "read LocalImageFrame" should "work properly" in {
    val local = ImageFrame.read(resource.getFile).asInstanceOf[LocalImageFrame]
    local.array.length should be(1)
    assert(local.array(0).uri.endsWith("000025.jpg"))
    assert(local.array(0).bytes.length == 95959)
    local.array(0).opencvMat().shape() should be((375, 500, 3))
  }

  "LocalImageFrame toDistributed" should "work properly" in {
    val local = ImageFrame.read(resource.getFile).asInstanceOf[LocalImageFrame]
    local.array.foreach(x => println(x.uri, x.bytes.length))
    val imageFeature = local.toDistributed(sc).rdd.first()
    assert(imageFeature.uri.endsWith("000025.jpg"))
    assert(imageFeature.bytes.length == 95959)
    imageFeature.opencvMat().shape() should be((375, 500, 3))
  }

  "read DistributedImageFrame" should "work properly" in {
    val distributed = ImageFrame.read(resource.getFile, sc)
      .asInstanceOf[DistributedImageFrame]
    val imageFeature = distributed.rdd.first()
    assert(imageFeature.uri.endsWith("000025.jpg"))
    assert(imageFeature.bytes.length == 95959)
    imageFeature.opencvMat().shape() should be((375, 500, 3))
  }

  "SequenceFile write and read" should "work properly" in {
    val tmpFile = Files.createTempDir()
    val dir = tmpFile.toString + "/parque"
    ImageFrame.writeParquet(resource.getFile, dir, sqlContext)

    val distributed = ImageFrame.readParquet(dir, sqlContext)
    val imageFeature = distributed.rdd.first()
    assert(imageFeature.uri.endsWith("000025.jpg"))
    assert(imageFeature.bytes.length == 95959)
    FileUtils.deleteDirectory(tmpFile)
  }

  "read local" should "work" in {
    val images = ImageFrame.read(resource.getFile).asInstanceOf[LocalImageFrame]
    images.array.length should be (1)

    val images2 = ImageFrame.read(resource.getFile + "*.jpg").asInstanceOf[LocalImageFrame]
    images2.array.length should be (1)

    val images3 = ImageFrame.read(resource.getFile + "000025.jpg").asInstanceOf[LocalImageFrame]
    images3.array.length should be (1)

    val images4 = ImageFrame.read(resource.getFile + "0000251.jpg").asInstanceOf[LocalImageFrame]
    images4.array.length should be (0)
  }
}
