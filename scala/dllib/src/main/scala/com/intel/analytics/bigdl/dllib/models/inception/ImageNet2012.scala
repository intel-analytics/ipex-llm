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
import com.intel.analytics.bigdl.dataset.image.{BGRImgCropper, BGRImgNormalizer, BytesToBGRImg, CropCenter, MTLabeledBGRImgToBatch, HFlip => DatasetHFlip}
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
    DataSet.SeqFileFolder.files(path, sc, classNumber).transform(
      MTLabeledBGRImgToBatch[ByteRecord](
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = (BytesToBGRImg() -> BGRImgCropper(imageSize, imageSize)
          -> DatasetHFlip(0.5) -> BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225))
      ))
  }

  def rdd(path: String, batchSize: Int, sc: SparkContext, imageSize : Int)
  : DataSet[MiniBatch[Float]] = {
    val imageFrame = DataSet.SeqFileFolder.filesToImageFrame(path, sc, 1000)
    val transfomer = PixelBytesToMat() ->
      Resize(256, 256) ->
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
     DataSet.SeqFileFolder.files(path, sc, classNumber).transform(
       MTLabeledBGRImgToBatch[ByteRecord](
         width = imageSize,
         height = imageSize,
         batchSize = batchSize,
         transformer = (BytesToBGRImg() -> BGRImgCropper(imageSize, imageSize, CropCenter)
           -> DatasetHFlip(0.5) -> BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225))
       ))
   }

  def rdd(path: String, batchSize: Int, sc: SparkContext, imageSize : Int)
  : DataSet[MiniBatch[Float]] = {
    val imageFrame = DataSet.SeqFileFolder.filesToImageFrame(path, sc, 1000)
    val transfomer = PixelBytesToMat() ->
      Resize(256, 256) ->
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

