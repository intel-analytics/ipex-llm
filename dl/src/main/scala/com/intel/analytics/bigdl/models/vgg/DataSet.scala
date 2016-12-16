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
package com.intel.analytics.bigdl.models.vgg

import java.nio.file.Path

import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.SparkContext

object DataSet {
  def localDataSet(imagesFile: Path, looped: Boolean, batchSize : Int)
  : LocalDataSet[Batch[Float]] = {
    val ds = LocalImageFiles.localBytesDataSet(imagesFile, looped, 32)
    val toImage = SampleToRGBImg()
    val normalizer = RGBImgNormalizer(ds -> toImage)
    val toBatch = new RGBImgToBatch(batchSize)
    ds -> toImage -> normalizer -> toBatch
  }

  def distributedDataSet(imagesFile: Path, looped: Boolean, sc: SparkContext,
    partitionNum: Int, batchSize : Int): DistributedDataSet[Batch[Float]] = {
    val ds = LocalImageFiles.distriDataSet(imagesFile, looped, sc, partitionNum, 32)

    val toImage = SampleToRGBImg()
    val normalizer = RGBImgNormalizer(0.5, 0.5, 0.5, 1.0, 1.0, 1.0)
    val toTensor = new RGBImgToBatch(batchSize)
    ds -> toImage -> normalizer -> toTensor
  }
}
