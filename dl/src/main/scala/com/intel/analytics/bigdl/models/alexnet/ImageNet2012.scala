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

import com.intel.analytics.bigdl.dataset.{Batch, DataSet, LocalDataSet, SeqFileLocalPath}
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine

object ImageNet2012 {
  def apply(path: Path, imageSize: Int, batchSize: Int, size: Int)
  : LocalDataSet[Batch[Float]] = {
    DataSet.SequenceFolder.paths(path, size)
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
  }
}
