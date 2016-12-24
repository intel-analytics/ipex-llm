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

import java.nio.file.{Path, Paths}

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext

object ImageNet2012 {
  def apply(
    path: String,
    sc: Option[SparkContext],
    imageSize: Int,
    batchSize: Int,
    nodeNumber: Int,
    coresPerNode: Int,
    classNumber: Int,
    size: Int
  )
  : DataSet[Batch[Float]] = {
    (if (sc.isDefined) {
      DataSet.SequenceFolder.files(path, sc.get, classNumber, nodeNumber).transform(
        MTLabeledBGRImgToBatch[Sample](
          width = imageSize,
          height = imageSize,
          batchSize = batchSize,
          transformer = (SampleToBGRImg() -> BGRImgCropper(imageSize, imageSize)
            -> HFlip(0.5) -> BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225))
        ))
    } else {
      DataSet.SequenceFolder.paths(Paths.get(path), size)
        .transform(
          MTLabeledBGRImgToBatch(
            width = imageSize,
            height = imageSize,
            batchSize = batchSize,
            transformer = (LocalSeqFileToBytes() -> SampleToBGRImg() ->
              BGRImgCropper(cropWidth = imageSize, cropHeight = imageSize) -> HFlip(0.5) ->
              BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
              )
          )
        )
    })
  }
}
