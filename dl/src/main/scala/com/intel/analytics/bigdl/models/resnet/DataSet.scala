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

import com.intel.analytics.bigdl.dataset.{Batch, SeqFileLocalPath, LocalDataSet}
import com.intel.analytics.bigdl.dataset.image._

trait DataSet {
  def localTrainDataSet(path: Path, imageSize: Int, batchSize: Int, parallel: Int, looped: Boolean)
  : LocalDataSet[Batch[Float]]
  def localValDataSet(path: Path, imageSize: Int, batchSize: Int, parallel: Int, looped: Boolean)
  : LocalDataSet[Batch[Float]]
}

object ImagenetDataSet extends DataSet {
  def localTrainDataSet(path: Path, imageSize: Int, batchSize: Int, parallel: Int, looped: Boolean)
  : LocalDataSet[Batch[Float]] = {
    val ds = SequenceFiles.localFiles(path, 1281167, looped)
    val fileTransformer = LocalSeqFileToBytes()
    val arrayToImage = SampleToRGBImg()
    val cropper = RGBImgCropper(cropWidth = imageSize, cropHeight = imageSize)
    val normalizer = RGBImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    val flipper = HFlip(0.5)
    val colorJitter = ColorJitter()
    val lighting = Lighting()
    val multiThreadToTensor = MTLabeledRGBImgToTensor[SeqFileLocalPath](
      width = imageSize,
      height = imageSize,
      threadNum = parallel,
      batchSize = batchSize,
      transformer = fileTransformer -> arrayToImage -> cropper -> colorJitter -> lighting -> normalizer
        -> flipper
    )
    ds -> multiThreadToTensor
  }

  def localValDataSet(path: Path, imageSize: Int, batchSize: Int, parallel: Int, looped: Boolean)
  : LocalDataSet[Batch[Float]] = {
    val ds = SequenceFiles.localFiles(path, 1281167, looped)
    val fileTransformer = LocalSeqFileToBytes()
    val arrayToImage = SampleToRGBImg()
    val cropper = RGBImgCropper(cropWidth = imageSize, cropHeight = imageSize)
    val normalizer = RGBImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    val multiThreadToTensor = MTLabeledRGBImgToTensor[SeqFileLocalPath](
      width = imageSize,
      height = imageSize,
      threadNum = parallel,
      batchSize = batchSize,
      transformer = fileTransformer -> arrayToImage -> normalizer -> cropper
    )
    ds -> multiThreadToTensor
  }
}

object Cifar10DataSet extends DataSet {
  def localTrainDataSet(path : Path, imageSize : Int, batchSize : Int, parallel: Int, looped : Boolean)
  : LocalDataSet[Batch[Float]] = {
    val ds = SequenceFiles.localFiles(path, 50000, looped)
    val fileTransformer = LocalSeqFileToBytes()
    val arrayToImage = SampleToRGBImg()
    val rdmCropper = RGBImgRdmCropper(cropWidth = imageSize, cropHeight = imageSize, padding = 4)
    val normalizer = RGBImgNormalizer(ds -> fileTransformer -> arrayToImage)
    val flipper = HFlip(0.5)
    val multiThreadToTensor = MTLabeledRGBImgToTensor[SeqFileLocalPath](
      width = imageSize,
      height = imageSize,
      threadNum = parallel,
      batchSize = batchSize,
      transformer = fileTransformer -> arrayToImage -> normalizer -> flipper -> rdmCropper
    )
    ds -> multiThreadToTensor
  }
  def localValDataSet(path : Path, imageSize : Int, batchSize : Int, parallel: Int, looped : Boolean)
  : LocalDataSet[Batch[Float]] = {
    val ds = SequenceFiles.localFiles(path, 10000, looped)
    val fileTransformer = LocalSeqFileToBytes()
    val arrayToImage = SampleToRGBImg()
    val normalizer = RGBImgNormalizer(ds -> fileTransformer -> arrayToImage)
    val multiThreadToTensor = MTLabeledRGBImgToTensor[SeqFileLocalPath](
      width = imageSize,
      height = imageSize,
      threadNum = parallel,
      batchSize = batchSize,
      transformer = fileTransformer -> arrayToImage -> normalizer
    )
    ds -> multiThreadToTensor
  }

}
