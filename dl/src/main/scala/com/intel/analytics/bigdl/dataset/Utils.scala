/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Engine

import scala.collection.Iterator

object Utils {

  def getBatchSize(totalBatch : Int): Int = {
    if (Engine.nodeNumber().isDefined) {
      val nodeNumber = Engine.nodeNumber().get
      val coreNumber = Engine.coreNumber()
      require(totalBatch % (nodeNumber * coreNumber) == 0
        , s"total batch size($totalBatch) can't be divided by node number($nodeNumber) * " +
          s"core number($coreNumber), please change your batch size")
      require(totalBatch >= nodeNumber * coreNumber * 2
        , s"total batch size($totalBatch) should be at least two times of node number" +
          s"($nodeNumber) * core number($coreNumber), please change your batch size")
      totalBatch / nodeNumber
    } else {
      totalBatch
    }
  }

}

object SampleToBatch {
  def apply(batchSize: Int): SampleToBatch
  = new SampleToBatch(batchSize)
}

class SampleToBatch(totalBatch: Int)
  extends Transformer[Sample, MiniBatch[Float]] {

  override def apply(prev: Iterator[Sample]): Iterator[MiniBatch[Float]] = {
    new Iterator[MiniBatch[Float]] {
      private val featureTensor: Tensor[Float] = Tensor[Float]()
      private val labelTensor: Tensor[Float] = Tensor[Float]()
      private var featureData: Array[Float] = null
      private var labelData: Array[Float] = null
      private val batchSize = Utils.getBatchSize(totalBatch)
      private var dimension: (Array[Int], Array[Int]) = null
      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[Float] = {
        if (prev.hasNext) {
          var i = 0
          while (i < batchSize && prev.hasNext) {
            val smpl = prev.next()
            val featureLength = smpl.getFeature().size.reduceLeft(_*_)
           val labelLength = smpl.getLabel().size.reduceLeft(_*_)
            dimension = (smpl.getFeature().size, smpl.getLabel().size)
            if (featureData == null || featureData.length < batchSize * featureLength) {
              featureData = new Array[Float](batchSize * featureLength)
            }
            if (labelData == null || labelData.length < batchSize * labelLength) {
              labelData = new Array[Float](batchSize * labelLength)
            }
            smpl.copyToFeature(featureData, i*featureLength, featureLength)
            smpl.copyToLabel(labelData, i*labelLength, labelLength)
            i += 1
          }

          featureTensor.set(Storage[Float](featureData),
            storageOffset = 1, sizes = Array(Array(i), dimension._1).flatten)
          labelTensor.set(Storage[Float](labelData),
            storageOffset = 1, sizes = Array(Array(i), dimension._2).flatten)

          MiniBatch(featureTensor, labelTensor)
        } else {
          null
        }
      }
    }
  }
}
