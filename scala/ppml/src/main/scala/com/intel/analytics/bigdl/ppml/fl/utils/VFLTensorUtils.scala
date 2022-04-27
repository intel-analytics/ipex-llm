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

package com.intel.analytics.bigdl.ppml.fl.utils

import com.intel.analytics.bigdl.dllib.feature.dataset.{DataSet, LocalDataSet, MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.dllib.tensor.Tensor

/**
 * In Vertical Federated Learning, some clients do not have labels
 * This util is for processing training data and labels in this special case for VFL
 */
object VFLTensorUtils {
  def featureLabelToMiniBatch(x: Tensor[Float],
                              y: Tensor[Float],
                              batchSize: Int): LocalDataSet[MiniBatch[Float]] = {
    if (x == null) return null
    val dataSize = x.size()(0)
    val sample = (0 until dataSize).map(index => {
      if (y != null) {
        Sample(x.select(1, index + 1), y.select(1, index + 1))
      } else {
        Sample(x.select(1, index + 1))
      }
    })
    (DataSet.array(sample.toArray)
      -> SampleToMiniBatch(batchSize, parallelizing = false)).toLocal()
  }
}
