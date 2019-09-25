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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.utils.{Engine, Table}
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class MTImageFeatureToBatchSpec extends FlatSpec with Matchers with BeforeAndAfter {

  var sc: SparkContext = null
  before {
    val conf = Engine.createSparkConf().setAppName("MTImageFeatureToBatchSpec")
      .setMaster("local[2]")
    sc = new SparkContext(conf)
    Engine.init
  }

  after {
    if (null != sc) sc.stop()
  }

  // todo: There is a race-condition bug in MTImageFeatureToBatch
  /*
  "MTImageFeatureToBatch classification" should "work well" in {
    //
    val imgData = (0 to 1000).map(idx => (idx to (idx + 10*10*3)).map(_.toFloat).toArray)
      .map(arr => {
        val imf = ImageFeature()
        imf(ImageFeature.floats) = arr
        val lab = Tensor[Float](Array(arr(0)), Array(1))
        imf(ImageFeature.label) = lab
        imf(ImageFeature.originalSize) = (10, 10, 3)
        imf
      })
    val transformer = MTImageFeatureToBatch(10, 10, 19, new FeatureTransformer {}, toRGB = false)
    val miniBatch = transformer(imgData.toIterator)
    // val imgCheck = new Array[Boolean](1000)
    miniBatch
      .take(5)
      // .take(1000 / 19)
      .foreach(batch => {
      (batch.size() <= 19) should be (true)
      val input = batch.getInput().asInstanceOf[Tensor[Float]]
      val target = batch.getTarget().asInstanceOf[Tensor[Float]]
      input.size() should be (Array(batch.size(), 3, 10, 10))
      target.size() should be (Array(batch.size()))
      for(i <- 1 to batch.size()) {
        // R
        val idx = input.valueAt(i, 1, 1, 1).toInt
        val G = input.valueAt(i, 2, 1, 1).toInt
        val B = input.valueAt(i, 3, 1, 1).toInt
        idx should be (G - 1)
        B should be (G + 1)
        input.valueAt(i, 3, 10, 10) should be((idx.toFloat + 10 * 10 * 3 - 1) +- 0.000001f)
        target.valueAt(i) should be (idx.toFloat)
        /* imgCheck(idx) should be (false)
        imgCheck(idx) = true */
      }

    })

  }

  "MTImageFeatureToBatch with ROI" should "work well" in {
    //
    val imgData = (0 to 1000).map(idx => (idx to (idx + 10*10*3)).map(_.toFloat).toArray)
      .map(arr => {
        val imf = ImageFeature()
        imf(ImageFeature.floats) = arr
        imf(ImageFeature.label) = RoiLabel(
          Tensor(new Array[Float](2), Array(2)),
          Tensor(new Array[Float](2*4), Array(2, 4)),
          Array(Tensor[Float](), Tensor[Float]())
        )
        imf(RoiLabel.ISCROWD) = Tensor(Array(0f, 1f), Array(2))
        imf(ImageFeature.originalSize) = (10, 10, 3)
        imf
      })
    val transformer = MTImageFeatureToBatch(10, 10, 19, new FeatureTransformer {},
      toRGB = false, extractRoi = true)
    val miniBatch = transformer(imgData.toIterator)
    // val imgCheck = new Array[Boolean](1000)
    miniBatch
      .take(5)
      // .take(1000 / 19)
      .foreach(batch => {
      (batch.size() <= 19) should be (true)
      val target = batch.getTarget().asInstanceOf[Table]
      target.length() should be (batch.size())
      for(i <- 1 to batch.size()) {
        val t = target(i).asInstanceOf[Table]
        t[Tensor[Float]](RoiLabel.ISCROWD) should be (Tensor(Array(0f, 1f), Array(2)))
        t[(Int, Int, Int)](RoiLabel.ORIGSIZE) should be((10, 10, 3))
        t[Tensor[Float]](RoiLabel.BBOXES).size() should be (Array(2, 4))
        t[Tensor[Float]](RoiLabel.CLASSES).size() should be (Array(2))
        t[Array[Tensor[Float]]](RoiLabel.MASKS).length should be (2)
      }

    })

  } */

}
