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
package com.intel.analytics.bigdl.models.inception

import java.nio.file.Paths

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image.{BGRImgCropper, BGRImgNormalizer, BytesToBGRImg, CropCenter, CropRandom, MTLabeledBGRImgToBatch, HFlip => DatasetHFlip}
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

object ImageNet2012 {
  def apply(
    path : String,
    sc: SparkContext,
    imageSize : Int,
    batchSize : Int,
    nodeNumber: Int,
    coresPerNode: Int,
    classNumber: Int
  )
  : DataSet[MiniBatch[Float]] = {
    DataSet.SeqFileFolder.filesToImageFeatureDataset(path, sc, classNumber).transform(
      MTImageFeatureToBatch(
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = PixelBytesToMat() ->
          Resize(256, 256) ->
          RandomCropper(224, 224, true, CropRandom) ->
          ChannelNormalize(123, 117, 104) ->
          MatToTensor[Float](), toRGB = false
      )
    )
  }

  def rdd(path: String, batchSize: Int, sc: SparkContext, imageSize : Int)
  : DataSet[MiniBatch[Float]] = {
    val imageFrame = DataSet.SeqFileFolder.filesToImageFrame(path, sc, 1000)
    val transfomer = PixelBytesToMat() ->
      RandomCrop(imageSize, imageSize) ->
      RandomTransformer(HFlip(), 0.5) ->
      ChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      MatToTensor[Float]() ->
      ImageFrameToSample[Float](targetKeys = Array(ImageFeature.label)) ->
      ImageFeatureToMiniBatch[Float](batchSize)
    val data = DataSet.imageFrame(imageFrame).transform(transfomer)
    data
  }
}

object ImageNet2012Val {
   def apply(
     path : String,
     sc: SparkContext,
     imageSize : Int,
     batchSize : Int,
     nodeNumber: Int,
     coresPerNode: Int,
     classNumber: Int
   )
   : DataSet[MiniBatch[Float]] = {

     DataSet.SeqFileFolder.filesToImageFeatureDataset(path, sc, 1000).transform(
       MTImageFeatureToBatch(
         width = imageSize,
         height = imageSize,
         batchSize = batchSize,
         transformer = PixelBytesToMat() ->
           Resize(256, 256) ->
           RandomCropper(224, 224, false, CropCenter) ->
           ChannelNormalize(123, 117, 104) ->
           MatToTensor[Float](), toRGB = false
       )
     )
   }

  def rdd(path: String, batchSize: Int, sc: SparkContext, imageSize : Int)
  : DataSet[MiniBatch[Float]] = {
    val imageFrame = DataSet.SeqFileFolder.filesToImageFrame(path, sc, 1000)
    val transfomer = PixelBytesToMat() ->
      CenterCrop(imageSize, imageSize) ->
      RandomTransformer(HFlip(), 0.5) ->
      ChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      MatToTensor[Float]() ->
      ImageFrameToSample[Float](targetKeys = Array(ImageFeature.label)) ->
      ImageFeatureToMiniBatch[Float](batchSize)
    val data = DataSet.imageFrame(imageFrame).transform(transfomer)
    data
  }

 }

