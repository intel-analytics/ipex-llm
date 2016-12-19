/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.models.resnet

import java.nio.file.{Path, Paths}

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image._
import org.apache.spark.SparkContext

trait ResNetDataSet {
  def localTrainDataSet(path: Path, batchSize: Int, size: Int)
  : LocalDataSet[Batch[Float]]
  def localValDataSet(path: Path, batchSize: Int, size: Int)
  : LocalDataSet[Batch[Float]]
  def distributedValDataSet(path: Path, sc: SparkContext, partitionNum: Int, imageSize: Int, batchSize: Int)
  : DistributedDataSet[Batch[Float]]
  def distributedTrainDataSet(path: Path, sc: SparkContext, partitionNum: Int, imageSize: Int, batchSize: Int)
    : DistributedDataSet[Batch[Float]]
}

/*object ImagenetDataSet extends ResNetDataSet {

  override def localTrainDataSet(path: Path, batchSize: Int, size: Int)
  : LocalDataSet[Batch[Float]] = {
    DataSet.SequenceFolder.paths(path, size)
      .transform(
        MTLabeledRGBImgToBatch(
          width = size,
          height = size,
          batchSize = batchSize,
          transformer = (LocalSeqFileToBytes() -> SampleToRGBImg() ->
            RGBImgCropper(cropWidth = 224, cropHeight = 224) ->
            ColorJitter() -> Lighting() ->
            RGBImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225) ->
            HFlip(0.5)))
      )
  }

  override def localValDataSet(path: Path, batchSize: Int, size: Int)
  : LocalDataSet[Batch[Float]] = {
    DataSet.SequenceFolder.paths(path, size)
      .transform(
        MTLabeledRGBImgToBatch(
          width = size,
          height = size,
          batchSize = batchSize,
          transformer = (LocalSeqFileToBytes() -> SampleToRGBImg() ->
            RGBImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225) ->
            RGBImgCropper(cropWidth = 224, cropHeight = 224))
        )
      )
  }

  override def distributedValDataSet(path: Path, sc: SparkContext, partitionNum: Int, imageSize: Int, batchSize: Int)
  : DistributedDataSet[Batch[Float]] = {
    val ds = LocalImageFiles.distriDataSet(imagesFile, looped, sc, partitionNum, 224)
    val fileTransformer = LocalSeqFileToBytes()
    val arrayToImage = SampleToRGBImg()
    val cropper = RGBImgCropper(cropWidth = 224, cropHeight = 224)
    val normalizer = RGBImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    val toBatch = new RGBImgToBatch(batchSize)
    ds -> arrayToImage -> normalizer -> cropper -> toBatch
  }
  override def distributedTrainDataSet(path: Path, sc: SparkContext, partitionNum: Int, imageSize: Int, batchSize: Int)
  : DistributedDataSet[Batch[Float]] = {
    val ds = LocalImageFiles.distriDataSet(imagesFile, looped, sc, partitionNum, 224)
    val arrayToImage = SampleToRGBImg()
    val cropper = RGBImgCropper(cropWidth = 224, cropHeight = 224)
    val normalizer = RGBImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    val flipper = HFlip(0.5)
    val colorJitter = ColorJitter()
    val lighting = Lighting()
    val toBatch = new RGBImgToBatch(batchSize)
    ds -> arrayToImage -> cropper -> colorJitter -> lighting -> normalizer -> toBatch
  }

}*/

object Cifar10DataSet extends ResNetDataSet {

  override def localTrainDataSet(path: Path, batchSize: Int, size: Int)
  : LocalDataSet[Batch[Float]] = {

    DataSet.ImageFolder.images(path, size)
      .transform(RGBImgNormalizer(125.3, 123.0, 113.9, 63.0, 62.1, 66.7))
      .transform(HFlip(0.5))
      .transform(RGBImgRdmCropper(cropWidth = 32, cropHeight = 32, padding = 4))
      .transform(RGBImgToBatch(batchSize))
  }

  override def localValDataSet(path: Path, batchSize: Int, size: Int)
  : LocalDataSet[Batch[Float]] = {

    DataSet.ImageFolder.images(path, size)
      .transform(RGBImgNormalizer(125.3, 123.0, 113.9, 63.0, 62.1, 66.7))
      .transform(RGBImgToBatch(batchSize))
  }

  override def distributedValDataSet(path: Path, sc: SparkContext, partitionNum: Int, imageSize: Int, batchSize: Int)
  : DistributedDataSet[Batch[Float]] = {

    DataSet.ImageFolder.images(path, sc, partitionNum, imageSize)
      .transform(RGBImgNormalizer(125.3, 123.0, 113.9, 63.0, 62.1, 66.7))
      .transform(RGBImgToBatch(batchSize))
  }

  override def distributedTrainDataSet(path: Path, sc: SparkContext, partitionNum: Int, imageSize: Int, batchSize: Int)
  : DistributedDataSet[Batch[Float]] = {

    DataSet.ImageFolder.images(path, sc, partitionNum, imageSize)
      .transform(RGBImgNormalizer(125.3, 123.0, 113.9, 63.0, 62.1, 66.7))
      .transform(HFlip(0.5))
      .transform(RGBImgRdmCropper(cropWidth = 32, cropHeight = 32, padding = 4))
      .transform(RGBImgToBatch(batchSize))
  }
}
