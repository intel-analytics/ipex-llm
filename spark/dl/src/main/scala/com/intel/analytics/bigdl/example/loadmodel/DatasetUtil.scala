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

package com.intel.analytics.bigdl.example.loadmodel

import java.nio.file.{Paths, Path}

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image.{BGRImgCropper, BGRImgNormalizer, BGRImgPixelNormalizer, BytesToBGRImg, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.File
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


object AlexNetPreprocessor {
  val imageSize = 227

  def apply(path: String, batchSize: Int, meanFile: String, sc: SparkContext)
  : DataSet[MiniBatch[Float]] = {
    val means = File.load[Tensor[Float]](meanFile)
    DataSet.SeqFileFolder.files(path, sc, classNum = 1000) ->
      // do not normalize the pixel values to [0, 1]
      BytesToBGRImg(normalize = 1f) ->
      BGRImgPixelNormalizer(means) -> BGRImgCropper(imageSize, imageSize, CropCenter) ->
      BGRImgToBatch(batchSize, toRGB = false)
  }

  def rdd(path: String, batchSize: Int, meanFile: String, sc: SparkContext)
  : RDD[Sample[Float]] = {
    val means = File.load[Tensor[Float]](meanFile)
    val dataSet = DataSet.SeqFileFolder.filesToRdd(path, sc, 1000)
      // do not normalize the pixel values to [0, 1]
    val transfomer = BytesToBGRImg(normalize = 1f) ->
      BGRImgPixelNormalizer(means) -> BGRImgCropper(imageSize, imageSize, CropCenter) ->
      BGRImgToSample(toRGB = false)
    transfomer(dataSet)
  }
}

object InceptionPreprocessor {
  val imageSize = 224

  def apply(path: String, batchSize: Int, sc: SparkContext)
  : DataSet[MiniBatch[Float]] = {
    DataSet.SeqFileFolder.files(path, sc, classNum = 1000) ->
      BytesToBGRImg(normalize = 1f) ->
      BGRImgCropper(imageSize, imageSize, CropCenter) ->
      BGRImgNormalizer(123, 117, 104, 1, 1, 1) ->
      BGRImgToBatch(batchSize, toRGB = false)
  }

  def rdd(path: String, batchSize: Int, sc: SparkContext)
  : RDD[Sample[Float]] = {
    val dataSet = DataSet.SeqFileFolder.filesToRdd(path, sc, classNum = 1000)
    val transfomer = BytesToBGRImg(normalize = 1f) ->
      BGRImgCropper(imageSize, imageSize, CropCenter) ->
      BGRImgNormalizer(123, 117, 104, 1, 1, 1) -> BGRImgToSample(toRGB = false)
    transfomer(dataSet)
  }
}

object ResNetPreprocessor {
  val imageSize = 224

  def apply(path: String, batchSize: Int, sc: SparkContext)
  : DataSet[MiniBatch[Float]] = {
    DataSet.SeqFileFolder.files(path.toString, sc, classNum = 1000) ->
      BytesToBGRImg() ->
      BGRImgCropper(cropWidth = imageSize, cropHeight = imageSize, CropCenter) ->
      BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225) ->
      BGRImgToBatch(batchSize)
  }

  def rdd(path: String, batchSize: Int, sc: SparkContext)
  : RDD[Sample[Float]] = {
    val dataSet = DataSet.SeqFileFolder.filesToRdd(path, sc, classNum = 1000)
    val transfomer = BytesToBGRImg() ->
      BGRImgCropper(cropWidth = imageSize, cropHeight = imageSize, CropCenter) ->
      BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225) -> BGRImgToSample()
    transfomer(dataSet)
  }
}
