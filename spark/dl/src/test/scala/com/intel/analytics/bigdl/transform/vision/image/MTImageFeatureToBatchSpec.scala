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

import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch}
import com.intel.analytics.bigdl.dataset.segmentation.RLEMasks
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
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

  "RoiImageFeatureToBatchWithMaxSize" should "work well" in {
    val imgData = (0 to 10).map(n => {
        val data = Tensor[Float](T(T(
          T(0.6336, 0.3563, 0.1053, 0.6912, 0.3791, 0.2707, 0.6270, 0.8446,
            0.2008, 0.3051, 0.6324, 0.4001, 0.6439, 0.2275, 0.8395, 0.6917),
          T(0.5191, 0.6917, 0.3929, 0.8765, 0.6981, 0.2679, 0.5423, 0.8095,
            0.1022, 0.0215, 0.1976, 0.3040, 0.3436, 0.0894, 0.5207, 0.9173),
          T(0.7829, 0.8493, 0.6865, 0.5468, 0.8769, 0.0055, 0.5274, 0.6638,
            0.5623, 0.6986, 0.9963, 0.9332, 0.3322, 0.2322, 0.7539, 0.1027),
          T(0.8297, 0.7903, 0.7254, 0.2109, 0.4015, 0.7729, 0.7242, 0.6415,
            0.0452, 0.5547, 0.7091, 0.8217, 0.6968, 0.7594, 0.3986, 0.5862),
          T(0.6075, 0.6215, 0.8243, 0.7298, 0.5886, 0.3655, 0.6750, 0.4722,
            0.1140, 0.2483, 0.8853, 0.4583, 0.2110, 0.8364, 0.2063, 0.4120),
          T(0.3350, 0.3226, 0.9264, 0.3657, 0.1387, 0.9268, 0.8490, 0.3405,
            0.1999, 0.2797, 0.8620, 0.2984, 0.1121, 0.9285, 0.3487, 0.1860),
          T(0.4850, 0.4671, 0.4069, 0.5200, 0.5928, 0.1164, 0.1781, 0.1367,
            0.0951, 0.8707, 0.8220, 0.3016, 0.8646, 0.9668, 0.7803, 0.1323),
          T(0.3663, 0.6169, 0.6257, 0.8451, 0.1146, 0.5394, 0.5738, 0.7960,
            0.4786, 0.6590, 0.5803, 0.0800, 0.0975, 0.1009, 0.1835, 0.5978)),
          T(T(0.6848, 0.7909, 0.0584, 0.5309, 0.5087, 0.3893, 0.5740, 0.8990,
          0.9438, 0.7067, 0.3653, 0.1513, 0.8279, 0.6395, 0.6875, 0.8965),
          T(0.8340, 0.4398, 0.5573, 0.2817, 0.1441, 0.7729, 0.0940, 0.9943,
            0.9369, 0.3792, 0.1262, 0.7556, 0.5480, 0.6573, 0.5901, 0.0393),
          T(0.1406, 0.5208, 0.4751, 0.6157, 0.5476, 0.9403, 0.0226, 0.6577,
            0.4105, 0.6823, 0.2789, 0.5607, 0.0228, 0.4178, 0.7816, 0.5339),
          T(0.6371, 0.0603, 0.3195, 0.6144, 0.2042, 0.1585, 0.1249, 0.9442,
            0.9533, 0.1570, 0.8457, 0.1685, 0.2243, 0.3009, 0.2149, 0.1328),
          T(0.7049, 0.6040, 0.5683, 0.3084, 0.2516, 0.1883, 0.0982, 0.7712,
            0.5637, 0.5811, 0.1678, 0.3323, 0.9634, 0.5855, 0.4315, 0.8492),
          T(0.6626, 0.1401, 0.7042, 0.3153, 0.6940, 0.5070, 0.6723, 0.6993,
            0.7467, 0.6185, 0.8907, 0.3982, 0.6435, 0.5429, 0.2580, 0.7538),
          T(0.3496, 0.3059, 0.1777, 0.7922, 0.9832, 0.5681, 0.6051, 0.1525,
            0.7647, 0.6433, 0.8886, 0.8596, 0.6976, 0.1161, 0.0092, 0.1787),
          T(0.0386, 0.8511, 0.4545, 0.1208, 0.2020, 0.7471, 0.7825, 0.3376,
            0.5597, 0.6067, 0.8809, 0.6917, 0.1960, 0.4223, 0.9569, 0.6081)),
          T(T(0.6848, 0.7909, 0.0584, 0.5309, 0.5087, 0.3893, 0.5740, 0.8990,
            0.9438, 0.7067, 0.3653, 0.1513, 0.8279, 0.6395, 0.6875, 0.8965),
            T(0.8340, 0.4398, 0.5573, 0.2817, 0.1441, 0.7729, 0.0940, 0.9943,
              0.9369, 0.3792, 0.1262, 0.7556, 0.5480, 0.6573, 0.5901, 0.0393),
            T(0.1406, 0.5208, 0.4751, 0.6157, 0.5476, 0.9403, 0.0226, 0.6577,
              0.4105, 0.6823, 0.2789, 0.5607, 0.0228, 0.4178, 0.7816, 0.5339),
            T(0.6371, 0.0603, 0.3195, 0.6144, 0.2042, 0.1585, 0.1249, 0.9442,
              0.9533, 0.1570, 0.8457, 0.1685, 0.2243, 0.3009, 0.2149, 0.1328),
            T(0.7049, 0.6040, 0.5683, 0.3084, 0.2516, 0.1883, 0.0982, 0.7712,
              0.5637, 0.5811, 0.1678, 0.3323, 0.9634, 0.5855, 0.4315, 0.8492),
            T(0.6626, 0.1401, 0.7042, 0.3153, 0.6940, 0.5070, 0.6723, 0.6993,
              0.7467, 0.6185, 0.8907, 0.3982, 0.6435, 0.5429, 0.2580, 0.7538),
            T(0.3496, 0.3059, 0.1777, 0.7922, 0.9832, 0.5681, 0.6051, 0.1525,
              0.7647, 0.6433, 0.8886, 0.8596, 0.6976, 0.1161, 0.0092, 0.1787),
            T(0.0386, 0.8511, 0.4545, 0.1208, 0.2020, 0.7471, 0.7825, 0.3376,
              0.5597, 0.6067, 0.8809, 0.6917, 0.1960, 0.4223, 0.9569, 0.6081))))
          .transpose(1, 2).transpose(2, 3).contiguous()

        val imf = ImageFeature()
        imf(ImageFeature.floats) = data.storage().array()
        imf(ImageFeature.label) = RoiLabel(
          Tensor(new Array[Float](2), Array(2)),
          Tensor(new Array[Float](2*4), Array(2, 4)),
          null
        )
        imf(RoiImageInfo.ISCROWD) = Tensor(Array(0f, 1f), Array(2))
        imf(ImageFeature.originalSize) = (8, 16, 3)
        imf
      }).toArray

    val transformer = RoiImageFeatureToBatch.withResize(3,
      new FeatureTransformer {}, toRGB = false, 10)
    val miniBatch = transformer(DataSet.array(imgData).data(false))

    val expectedOutput = Tensor[Float](T(T(
    T(0.6336, 0.3563, 0.1053, 0.6912, 0.3791, 0.2707, 0.6270, 0.8446, 0.2008, 0.3051,
      0.6324, 0.4001, 0.6439, 0.2275, 0.8395, 0.6917, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.5191, 0.6917, 0.3929, 0.8765, 0.6981, 0.2679, 0.5423, 0.8095, 0.1022, 0.0215,
      0.1976, 0.3040, 0.3436, 0.0894, 0.5207, 0.9173, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.7829, 0.8493, 0.6865, 0.5468, 0.8769, 0.0055, 0.5274, 0.6638, 0.5623, 0.6986,
      0.9963, 0.9332, 0.3322, 0.2322, 0.7539, 0.1027, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.8297, 0.7903, 0.7254, 0.2109, 0.4015, 0.7729, 0.7242, 0.6415, 0.0452, 0.5547,
      0.7091, 0.8217, 0.6968, 0.7594, 0.3986, 0.5862, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.6075, 0.6215, 0.8243, 0.7298, 0.5886, 0.3655, 0.6750, 0.4722, 0.1140, 0.2483,
      0.8853, 0.4583, 0.2110, 0.8364, 0.2063, 0.4120, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.3350, 0.3226, 0.9264, 0.3657, 0.1387, 0.9268, 0.8490, 0.3405, 0.1999, 0.2797,
      0.8620, 0.2984, 0.1121, 0.9285, 0.3487, 0.1860, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.4850, 0.4671, 0.4069, 0.5200, 0.5928, 0.1164, 0.1781, 0.1367, 0.0951, 0.8707,
      0.8220, 0.3016, 0.8646, 0.9668, 0.7803, 0.1323, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.3663, 0.6169, 0.6257, 0.8451, 0.1146, 0.5394, 0.5738, 0.7960, 0.4786, 0.6590,
      0.5803, 0.0800, 0.0975, 0.1009, 0.1835, 0.5978, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000)),
    T(T(0.6848, 0.7909, 0.0584, 0.5309, 0.5087, 0.3893, 0.5740, 0.8990, 0.9438, 0.7067,
      0.3653, 0.1513, 0.8279, 0.6395, 0.6875, 0.8965, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.8340, 0.4398, 0.5573, 0.2817, 0.1441, 0.7729, 0.0940, 0.9943, 0.9369, 0.3792,
      0.1262, 0.7556, 0.5480, 0.6573, 0.5901, 0.0393, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.1406, 0.5208, 0.4751, 0.6157, 0.5476, 0.9403, 0.0226, 0.6577, 0.4105, 0.6823,
      0.2789, 0.5607, 0.0228, 0.4178, 0.7816, 0.5339, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.6371, 0.0603, 0.3195, 0.6144, 0.2042, 0.1585, 0.1249, 0.9442, 0.9533, 0.1570,
      0.8457, 0.1685, 0.2243, 0.3009, 0.2149, 0.1328, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.7049, 0.6040, 0.5683, 0.3084, 0.2516, 0.1883, 0.0982, 0.7712, 0.5637, 0.5811,
      0.1678, 0.3323, 0.9634, 0.5855, 0.4315, 0.8492, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.6626, 0.1401, 0.7042, 0.3153, 0.6940, 0.5070, 0.6723, 0.6993, 0.7467, 0.6185,
      0.8907, 0.3982, 0.6435, 0.5429, 0.2580, 0.7538, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.3496, 0.3059, 0.1777, 0.7922, 0.9832, 0.5681, 0.6051, 0.1525, 0.7647, 0.6433,
      0.8886, 0.8596, 0.6976, 0.1161, 0.0092, 0.1787, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.0386, 0.8511, 0.4545, 0.1208, 0.2020, 0.7471, 0.7825, 0.3376, 0.5597, 0.6067,
      0.8809, 0.6917, 0.1960, 0.4223, 0.9569, 0.6081, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000)),
    T(T(0.6848, 0.7909, 0.0584, 0.5309, 0.5087, 0.3893, 0.5740, 0.8990, 0.9438, 0.7067,
      0.3653, 0.1513, 0.8279, 0.6395, 0.6875, 0.8965, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.8340, 0.4398, 0.5573, 0.2817, 0.1441, 0.7729, 0.0940, 0.9943, 0.9369, 0.3792,
      0.1262, 0.7556, 0.5480, 0.6573, 0.5901, 0.0393, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.1406, 0.5208, 0.4751, 0.6157, 0.5476, 0.9403, 0.0226, 0.6577, 0.4105, 0.6823,
      0.2789, 0.5607, 0.0228, 0.4178, 0.7816, 0.5339, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.6371, 0.0603, 0.3195, 0.6144, 0.2042, 0.1585, 0.1249, 0.9442, 0.9533, 0.1570,
      0.8457, 0.1685, 0.2243, 0.3009, 0.2149, 0.1328, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.7049, 0.6040, 0.5683, 0.3084, 0.2516, 0.1883, 0.0982, 0.7712, 0.5637, 0.5811,
      0.1678, 0.3323, 0.9634, 0.5855, 0.4315, 0.8492, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.6626, 0.1401, 0.7042, 0.3153, 0.6940, 0.5070, 0.6723, 0.6993, 0.7467, 0.6185,
      0.8907, 0.3982, 0.6435, 0.5429, 0.2580, 0.7538, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.3496, 0.3059, 0.1777, 0.7922, 0.9832, 0.5681, 0.6051, 0.1525, 0.7647, 0.6433,
      0.8886, 0.8596, 0.6976, 0.1161, 0.0092, 0.1787, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.0386, 0.8511, 0.4545, 0.1208, 0.2020, 0.7471, 0.7825, 0.3376, 0.5597, 0.6067,
      0.8809, 0.6917, 0.1960, 0.4223, 0.9569, 0.6081, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000),
    T(0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000))))

    miniBatch.foreach(batch => {
      (batch.size() <= 3) should be (true)
      val inputAll = batch.getInput().asInstanceOf[Table]
      val input = inputAll[Tensor[Float]](1)
      val target = batch.getTarget().asInstanceOf[Table]
      input.size() should be (Array(batch.size(), 3, 10, 20))
      target.length() should be (batch.size())
      for(i <- 1 to batch.size()) {
        val in = input.select(1, i)
        in should be(expectedOutput)
        val t = target(i).asInstanceOf[Table]
        t[Tensor[Float]](RoiImageInfo.ISCROWD) should be (Tensor(Array(0f, 1f), Array(2)))
        // t[(Int, Int, Int)](RoiImageInfo.ORIGSIZE) should be((8, 16, 3))
        t[Tensor[Float]](RoiImageInfo.BBOXES).size() should be (Array(2, 4))
        t[Tensor[Float]](RoiImageInfo.CLASSES).size() should be (Array(2))
      }
    })
  }

  "RoiMiniBatch" should "serialize well" in {
    def batch: RoiMiniBatch = RoiMiniBatch(
      Tensor[Float](),
      Array[RoiLabel](RoiLabel(Tensor[Float](), Tensor[Float]())),
      Array[Tensor[Float]](Tensor[Float]()),
      Tensor())
    val result = sc.parallelize(Array(batch, batch, batch, batch, batch), 3)
      .coalesce(2, true)
      .takeSample(false, 3).head
  }

  "MTImageFeatureToBatch classification" should "work well" in {
    val imgData = (0 to 1000).map(idx => (idx to (idx + 10*10*3)).map(_.toFloat).toArray)
      .map(arr => {
        val imf = ImageFeature()
        imf(ImageFeature.floats) = arr
        val lab = Tensor[Float](Array(arr(0)), Array(1))
        imf(ImageFeature.label) = lab
        imf(ImageFeature.originalSize) = (10, 10, 3)
        imf
      }).toArray
    val transformer = MTImageFeatureToBatch(10, 10, 19, new FeatureTransformer {}, toRGB = false)
    val miniBatch = transformer(DataSet.array(imgData).data(false))
    val imgCheck = new Array[Boolean](1001)
    miniBatch
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
        imgCheck(idx) should be (false)
        imgCheck(idx) = true
      }
    })
    imgCheck.count(!_) should be (0)

  }
  "MTImageFeatureToBatch without labels" should "work well" in {
    val imgData = (0 to 1000).map(idx => (idx to (idx + 10*10*3)).map(_.toFloat).toArray)
      .map(arr => {
        val imf = ImageFeature()
        imf(ImageFeature.floats) = arr
        imf(ImageFeature.originalSize) = (10, 10, 3)
        imf
      }).toArray
    val transformer = RoiImageFeatureToBatch(10, 10, 19, new FeatureTransformer {},
      toRGB = false)
    val miniBatch = transformer(DataSet.array(imgData).data(false))
    miniBatch.foreach(batch => {
      batch.asInstanceOf[RoiMiniBatch].target should be (null)
      batch.getInput().asInstanceOf[Table].get[Tensor[Float]](1) should not be (null)
    })

    val transformer2 = RoiImageFeatureToBatch.withResize(batchSize = 19,
      transformer = new FeatureTransformer {})
    val miniBatch2 = transformer(DataSet.array(imgData).data(false))
    miniBatch2.foreach(batch => {
      batch.asInstanceOf[RoiMiniBatch].target should be (null)
      batch.getInput().asInstanceOf[Table].get[Tensor[Float]](1) should not be (null)
    })

  }

  "MTImageFeatureToBatch with ROI" should "work well" in {
    val imgCheck = new Array[Boolean](1001)
    val imgData = (0 to 1000).map(idx => (idx to (idx + 10*10*3)).map(_.toFloat).toArray)
      .map(arr => {
        val imf = ImageFeature()
        imf(ImageFeature.floats) = arr
        imf(ImageFeature.label) = RoiLabel(
          Tensor(new Array[Float](2), Array(2)),
          Tensor(new Array[Float](2*4), Array(2, 4)),
          Array(new RLEMasks(Array(), 10, 10),
            new RLEMasks(Array(), 10, 10)
          )
        )
        imf(RoiImageInfo.ISCROWD) = Tensor(Array(0f, 1f), Array(2))
        imf(ImageFeature.originalSize) = (10, 10, 3)
        imf
      }).toArray
    val transformer = RoiImageFeatureToBatch(10, 10, 19, new FeatureTransformer {},
      toRGB = false)
    val miniBatch = transformer(DataSet.array(imgData).data(false))
    miniBatch
      .foreach(batch => {
      (batch.size() <= 19) should be (true)
      val target = batch.getTarget().asInstanceOf[Table]
      target.length() should be (batch.size())
      for(i <- 1 to batch.size()) {
        val t = target(i).asInstanceOf[Table]
        RoiImageInfo.getIsCrowd(t) should be (Tensor(Array(0f, 1f), Array(2)))
        RoiImageInfo.getImageInfo(t).size() should be(Array(4))
        RoiImageInfo.getBBoxes(t).size() should be (Array(2, 4))
        RoiImageInfo.getClasses(t).size() should be (Array(2))
        RoiImageInfo.getMasks(t).length should be (2)
        val idx = batch.getInput().asInstanceOf[Table].apply[Tensor[Float]](1)
          .valueAt(i, 1, 1, 1).toInt
        imgCheck(idx) should be (false)
        imgCheck(idx) = true
      }

    })
    imgCheck.count(!_) should be (0)

  }

  def arrayToTensor(a: Array[Float]): Tensor[Float] = Tensor[Float](a, Array(a.length))

  "RoiMiniBatch" should "correctly slice" in {
    val dummyBBox = Tensor[Float](Array(1f, 2, 3, 4), Array(1, 4))
    val roiLables = (0 until 100).map(i => {
      RoiLabel(arrayToTensor(Array(i.toFloat)), dummyBBox)
    }).toArray
    val isCrowds = (0 until 100).map(i => {
      arrayToTensor(Array(i.toFloat))
    }).toArray
    val b = RoiMiniBatch(
      arrayToTensor((1 to 100).toArray.map(_.toFloat)),
      roiLables,
      isCrowds,
      arrayToTensor((1 to 100).toArray.map(_.toFloat))
    )

    val s1 = b.slice(3, 20)

    def checkSlice(s1: MiniBatch[Float], start: Int, len: Int,
      checkTarget: Boolean = true, checkImgInfo: Boolean = true): Unit = {

      if (checkImgInfo) {
        val input = s1.getInput().toTable
        val imgData = input[Tensor[Float]](1)
        imgData.nElement() should be(len)
        imgData.valueAt(1) should be(start.toFloat)
        imgData.valueAt(len) should be(start.toFloat + len - 1)

        val imgInfo = input[Tensor[Float]](2)
        imgInfo.nElement() should be(len)
        imgInfo.valueAt(1) should be(start.toFloat)
        imgInfo.valueAt(len) should be(start.toFloat + len - 1)
      } else {
        val imgData = s1.getInput().toTensor[Float]
        imgData.nElement() should be(len)
        imgData.valueAt(1) should be(start.toFloat)
        imgData.valueAt(len) should be(start.toFloat + len - 1)
      }
      if (checkTarget) {
        val target = s1.getTarget().asInstanceOf[Table]
        target.length() should be (len)
        for (i <- 1 to target.length()) {
          val imgTarget = target[Table](i)
          RoiImageInfo.getBBoxes(imgTarget).size() should be (Array(1, 4))
          RoiImageInfo.getClasses(imgTarget).valueAt(1) should be (i.toFloat + start - 2)
          RoiImageInfo.getIsCrowd(imgTarget).nElement() should be (1)
          RoiImageInfo.getIsCrowd(imgTarget).valueAt(1) should be (i.toFloat + start - 2)
          RoiImageInfo.getImageInfo(imgTarget).value() should be (i.toFloat + start - 1)
        }
      }
    }

    checkSlice(s1, 3, 20)

    // check slice of slice
    val s2 = s1.slice(3, 10)
    checkSlice(s2, 5, 10)

    // this also checks empty target
    val b2 = RoiMiniBatch(
      arrayToTensor((1 to 100).toArray.map(_.toFloat)),
      null,
      isCrowds,
      arrayToTensor((1 to 100).toArray.map(_.toFloat))
    )
    checkSlice(b2.slice(12, 80).slice(2, 20),
      13, 20, false)

    val b3 = RoiMiniBatch(
      arrayToTensor((1 to 100).toArray.map(_.toFloat)),
      null,
      isCrowds
    )
    checkSlice(b3.slice(12, 80).slice(2, 20),
      13, 20, false, false)
  }

}
