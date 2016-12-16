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

import java.nio.file.Path

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image._
import org.apache.spark.SparkContext

trait DataSet {
  def localTrainDataSet(imagesFile: Path, looped: Boolean, batchSize : Int)
  : LocalDataSet[Batch[Float]]
  def localValDataSet(imagesFile: Path, looped: Boolean, batchSize : Int)
  : LocalDataSet[Batch[Float]]
  def distributedValDataSet(imagesFile: Path, looped: Boolean, sc: SparkContext,
                            partitionNum: Int, batchSize : Int): DistributedDataSet[Batch[Float]]
  def distributedTrainDataSet(imagesFile: Path, looped: Boolean, sc: SparkContext,
                              partitionNum: Int, batchSize : Int): DistributedDataSet[Batch[Float]]
}

object ImagenetDataSet extends DataSet {
  override def localTrainDataSet(imagesFile: Path, looped: Boolean, batchSize : Int)
  : LocalDataSet[Batch[Float]] = {
    val ds = LocalImageFiles.localBytesDataSet(imagesFile, looped, 32)
    val arrayToImage = SampleToRGBImg()
    val cropper = RGBImgCropper(cropWidth = 224, cropHeight = 224)
    val normalizer = RGBImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    val flipper = HFlip(0.5)
    val colorJitter = ColorJitter()
    val lighting = Lighting()
    val toBatch = new RGBImgToBatch(batchSize)
    ds -> arrayToImage -> cropper -> colorJitter -> lighting -> normalizer -> toBatch
  }

  override def localValDataSet(imagesFile: Path, looped: Boolean, batchSize : Int)
: LocalDataSet[Batch[Float]] = {
  val ds = LocalImageFiles.localBytesDataSet(imagesFile, looped, 32)
  val fileTransformer = LocalSeqFileToBytes()
    val arrayToImage = SampleToRGBImg()
    val cropper = RGBImgCropper(cropWidth = 224, cropHeight = 224)
    val normalizer = RGBImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
  val toBatch = new RGBImgToBatch(batchSize)
    ds -> arrayToImage -> normalizer -> cropper -> toBatch
  }

  override def distributedValDataSet(imagesFile: Path, looped: Boolean, sc: SparkContext,
                            partitionNum: Int, batchSize : Int): DistributedDataSet[Batch[Float]] = {
    val ds = LocalImageFiles.distriDataSet(imagesFile, looped, sc, partitionNum, 224)
    val fileTransformer = LocalSeqFileToBytes()
    val arrayToImage = SampleToRGBImg()
    val cropper = RGBImgCropper(cropWidth = 224, cropHeight = 224)
    val normalizer = RGBImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    val toBatch = new RGBImgToBatch(batchSize)
    ds -> arrayToImage -> normalizer -> cropper -> toBatch
  }
  override def distributedTrainDataSet(imagesFile: Path, looped: Boolean, sc: SparkContext,
                              partitionNum: Int, batchSize : Int): DistributedDataSet[Batch[Float]] = {
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

}

object Cifar10DataSet extends DataSet {

  override def localTrainDataSet(imagesFile: Path, looped: Boolean, batchSize : Int)
  : LocalDataSet[Batch[Float]] = {
    val ds = LocalImageFiles.localBytesDataSet(imagesFile, looped, 32)
    val arrayToImage = SampleToRGBImg()
    val rdmCropper = RGBImgRdmCropper(cropWidth = 32, cropHeight = 32, padding = 4)
    val normalizer = RGBImgNormalizer(125.3, 123.0, 113.9, 63.0, 62.1, 66.7)
    val flipper = HFlip(0.5)
    val toBatch = new RGBImgToBatch(batchSize)
    ds -> arrayToImage -> normalizer -> flipper -> rdmCropper -> toBatch
  }

  override def localValDataSet(imagesFile: Path, looped: Boolean, batchSize : Int)
  : LocalDataSet[Batch[Float]] = {
    val ds = LocalImageFiles.localBytesDataSet(imagesFile, looped, 32)
    val arrayToImage = SampleToRGBImg()
    val normalizer = RGBImgNormalizer(125.3, 123.0, 113.9, 63.0, 62.1, 66.7)
    val toBatch = new RGBImgToBatch(batchSize)
    ds -> arrayToImage -> normalizer -> toBatch
  }

  override def distributedValDataSet(imagesFile: Path, looped: Boolean, sc: SparkContext,
                         partitionNum: Int, batchSize : Int): DistributedDataSet[Batch[Float]] = {
    val ds = LocalImageFiles.distriDataSet(imagesFile, looped, sc, partitionNum, 32)

    val toImage = SampleToRGBImg()
    val normalizer = RGBImgNormalizer(125.3, 123.0, 113.9, 63.0, 62.1, 66.7)
    val toTensor = new RGBImgToBatch(batchSize)
    ds -> toImage -> normalizer -> toTensor
  }
  override def distributedTrainDataSet(imagesFile: Path, looped: Boolean, sc: SparkContext,
                            partitionNum: Int, batchSize : Int): DistributedDataSet[Batch[Float]] = {
    val ds = LocalImageFiles.distriDataSet(imagesFile, looped, sc, partitionNum, 32)
    val rdmCropper = RGBImgRdmCropper(cropWidth = 32, cropHeight = 32, padding = 4)
    val toImage = SampleToRGBImg()
    val normalizer = RGBImgNormalizer(125.3, 123.0, 113.9, 63.0, 62.1, 66.7)
    val flipper = HFlip(0.5)
    val toTensor = new RGBImgToBatch(batchSize)
    ds -> toImage -> normalizer -> flipper -> rdmCropper -> toTensor
  }

}
