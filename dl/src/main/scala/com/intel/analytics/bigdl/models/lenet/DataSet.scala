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

package com.intel.analytics.bigdl.models.lenet

import java.nio.ByteBuffer
import java.nio.file.{Files, Path}

import com.intel.analytics.bigdl.dataset.image.{LabeledBytesToGreyImg, LabeledGreyImageToTensor, LabeledGreyImgNormalizer}
import com.intel.analytics.bigdl.dataset.{CachedDistriDataSet, DistributedDataSet, LocalArrayDataSet, LocalDataSet}
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.SparkContext

/**
 * This data set will read MNIST files and cache all images in the memory. The MNIST file can be
 * download from http://yann.lecun.com/exdb/mnist/
 */
object DataSet {
  private def load(featureFile: Path, labelFile: Path): Array[(Float, Array[Byte])] = {
    val labelBuffer = ByteBuffer.wrap(Files.readAllBytes(labelFile))
    val featureBuffer = ByteBuffer.wrap(Files.readAllBytes(featureFile))
    val labelMagicNumber = labelBuffer.getInt()

    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()

    val result = new Array[(Float, Array[Byte])](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum))
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = (labelBuffer.get().toFloat + 1.0f, img)
      i += 1
    }

    result
  }

  def localDataSet(imagesFile: Path, labelsFile: Path, looped: Boolean, batchSize : Int)
  : LocalDataSet[(Tensor[Float], Tensor[Float])] = {
    val buffer = load(imagesFile, labelsFile)
    val ds = new LocalArrayDataSet[(Float, Array[Byte])](buffer, looped)
    val arrayByteToImage = LabeledBytesToGreyImg(28, 28)
    val normalizer = LabeledGreyImgNormalizer(ds -> arrayByteToImage)
    val toTensor = new LabeledGreyImageToTensor(batchSize)
    ds -> arrayByteToImage -> normalizer -> toTensor
  }

  def distributedDataSet(imagesFile: Path, labelsFile: Path, looped: Boolean, sc: SparkContext,
    partitionNum: Int, batchSize : Int): DistributedDataSet[(Tensor[Float], Tensor[Float])] = {
    val buffer = load(imagesFile, labelsFile)
    val ds = CachedDistriDataSet(buffer, sc, partitionNum, looped)
    val arrayByteToImage =
      LabeledBytesToGreyImg(28, 28)
    val normalizer = LabeledGreyImgNormalizer(0.5, 0.5)
    val toTensor = new LabeledGreyImageToTensor(batchSize)
    ds -> arrayByteToImage -> normalizer -> toTensor
  }
}
