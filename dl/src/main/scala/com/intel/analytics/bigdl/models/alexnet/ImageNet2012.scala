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
package com.intel.analytics.bigdl.models.alexnet

import java.nio.file.Path

import com.intel.analytics.bigdl.dataset.{DataSet, Batch, SeqFileLocalPath, LocalDataSet}
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.tensor.Tensor

object ImageNet2012 {
  def apply(path : Path, imageSize : Int, batchSize : Int, parallel: Int)
  : LocalDataSet[Batch[Float]] = {
    val ds = DataSet.SequenceFolder.paths(path, 1281167)
    val fileTransformer = LocalSeqFileToBytes()
    val arrayToImage = SampleToRGBImg()
    val cropper = RGBImgCropper(cropWidth = imageSize, cropHeight = imageSize)
    val normalizer = RGBImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    val flipper = HFlip(0.5)
    val multiThreadToTensor = MTLabeledRGBImgToTensor[SeqFileLocalPath](
      width = imageSize,
      height = imageSize,
      threadNum = parallel,
      batchSize = batchSize,
      transformer = fileTransformer -> arrayToImage -> cropper -> flipper -> normalizer
    )
    ds -> multiThreadToTensor
  }
}
