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

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image.{BGRImgCropper, BGRImgNormalizer, BGRImgPixelNormalizer, BytesToBGRImg, _}
import com.intel.analytics.bigdl.example.loadmodel.ModelValidator.{BigDlModel, ModelType, TorchModel}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.utils.File
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.io.Source


object AlexNetPreprocessor {
  val imageSize = 227

  def apply(path: String, batchSize: Int, meanFile: String, sc: SparkContext)
  : DataSet[MiniBatch[Float]] = {
    // 'meanFile' specify the path to the pixel level mean data, one line per pixel
    // following H * W * C order, 196608 in total (256 * 256 * 3)
    val means = createMeans(meanFile)
    DataSet.SeqFileFolder.files(path, sc, classNum = 1000) ->
      // do not normalize the pixel values to [0, 1]
      BytesToBGRImg(normalize = 1f, 256, 256) ->
      BGRImgPixelNormalizer(means) -> BGRImgCropper(imageSize, imageSize, CropCenter) ->
      BGRImgToBatch(batchSize, toRGB = false)
  }

  def rdd(path: String, batchSize: Int, meanFile: String, sc: SparkContext)
  : RDD[Sample[Float]] = {
    val means = createMeans(meanFile)
    val data = DataSet.SeqFileFolder.filesToImageFrame(path, sc, 1000)
      // do not normalize the pixel values to [0, 1]
    val transfomer = PixelBytesToMat() -> Resize(256, 256) ->
      PixelNormalizer(means.storage.array) -> CenterCrop(imageSize, imageSize) ->
      MatToTensor[Float]() -> ImageFrameToSample[Float](targetKeys = Array(ImageFeature.label))
    val imgFrame = data -> transfomer
    val validImageFeatures = imgFrame.toDistributed().rdd
    validImageFeatures.map(x => x[Sample[Float]](ImageFeature.sample))
  }

  def createMeans(meanFile : String) : Tensor[Float] = {
    val array = Source.fromFile(meanFile).getLines().map(_.toFloat).toArray
    Tensor[Float](array, Array(array.length))
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
    val data = DataSet.SeqFileFolder.filesToImageFrame(path, sc, 1000)
    val transfomer = PixelBytesToMat() -> Resize(256, 256) ->
      CenterCrop(imageSize, imageSize) -> ChannelNormalize(123, 117, 104) ->
      MatToTensor[Float]() -> ImageFrameToSample[Float](targetKeys = Array(ImageFeature.label))
    val imgFrame = transfomer(data)
    val validImageFeatures = imgFrame.toDistributed().rdd
    validImageFeatures.map(x => x[Sample[Float]](ImageFeature.sample))
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

  def rdd(path: String, batchSize: Int, sc: SparkContext, modelType : ModelType = TorchModel)
  : RDD[Sample[Float]] = {
    if (modelType == TorchModel) {
      val dataSet = DataSet.SeqFileFolder.filesToRdd(path, sc, classNum = 1000)
      val transfomer = BytesToBGRImg() ->
        BGRImgCropper(cropWidth = imageSize, cropHeight = imageSize, CropCenter) ->
        BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225) -> BGRImgToSample()
      transfomer(dataSet)
    } else if (modelType == BigDlModel) {
      val data = DataSet.SeqFileFolder.filesToImageFrame(path, sc, 1000)
      val transfomer = PixelBytesToMat() ->
        RandomResize(256, 256) ->
        CenterCrop(224, 224) ->
        ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
        MatToTensor[Float]() -> ImageFrameToSample[Float](targetKeys = Array(ImageFeature.label))
      val imgFrame = data -> transfomer
      val validImageFeatures = imgFrame.toDistributed().rdd
      validImageFeatures.map(x => x[Sample[Float]](ImageFeature.sample))
    } else {
      throw new IllegalArgumentException(s"${modelType} not recognized")
    }
  }
}

object VGGPreprocessor {
  val imageSize = 224

  def rdd(path: String, batchSize: Int, sc: SparkContext)
  : RDD[Sample[Float]] = {
    val data = DataSet.SeqFileFolder.filesToImageFrame(path, sc, 1000)
    val transfomer = PixelBytesToMat() -> Resize(256, 256) ->
      CenterCrop(imageSize, imageSize) -> ChannelNormalize(123, 117, 104) ->
      MatToTensor[Float]() -> ImageFrameToSample[Float](targetKeys = Array(ImageFeature.label))
    val imgFrame = transfomer(data)
    val validImageFeatures = imgFrame.toDistributed().rdd
    validImageFeatures.map(x => x[Sample[Float]](ImageFeature.sample))
  }
}

