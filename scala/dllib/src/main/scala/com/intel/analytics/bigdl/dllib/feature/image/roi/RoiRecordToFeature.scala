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

package com.intel.analytics.bigdl.dllib.feature.image.roi

import java.nio.ByteBuffer

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.dllib.feature.common.Preprocessing

import scala.collection.Iterator

// remove below line after migrate zoo.models
case class ByteRecord(var data: Array[Byte], path: String)

/**
 * Convert ROI image record to ImageFeature.
 *
 */
class RoiRecordToFeature(convertLabel: Boolean = false, outKey: String = ImageFeature.bytes)
  extends Preprocessing[ByteRecord, ImageFeature] {
  @transient var data: Array[Byte] = _
  @transient var gtClasses: Tensor[Float] = _
  @transient var gtBoxes: Tensor[Float] = _
  val feature = new ImageFeature()

  override def apply(prev: Iterator[ByteRecord]): Iterator[ImageFeature] = {
    prev.map(record => {
      feature.clear()
      val byteArray = record.data
      val byteBuffer = ByteBuffer.wrap(byteArray)
      val dataLen = byteBuffer.getInt
      val classLen = byteBuffer.getInt
      if (data == null || data.length < dataLen) {
        data = new Array[Byte](dataLen)
      }
      System.arraycopy(byteArray, 8, data, 0, dataLen)
      val target = if (convertLabel) {
        if (gtBoxes == null) {
          gtClasses = Tensor[Float]
          gtBoxes = Tensor[Float]
        }
        gtClasses.resize(2, classLen / 4)
        gtBoxes.resize(classLen / 4, 4)
        if (classLen > 0) {
          val gtClassesData = gtClasses.storage().array()
          val gtClassesOffset = gtClasses.storageOffset() - 1
          val gtBoxesData = gtBoxes.storage().array()
          val gtBoxesOffset = gtBoxes.storageOffset() - 1
          // label + difficult
          bytesToFloatTensor(byteArray, 8 + dataLen, classLen * 2, gtClassesData, gtClassesOffset)
          bytesToFloatTensor(byteArray, 8 + dataLen + classLen * 2, classLen * 4,
            gtBoxesData, gtBoxesOffset)
        }

        RoiLabel(gtClasses, gtBoxes)
      } else null
      feature(outKey) = data
      feature(ImageFeature.uri) = record.path
      feature(ImageFeature.label) = target
      feature
    })
  }


  def bytesToFloatTensor(src: Array[Byte], offset: Int, length: Int,
    target: Array[Float], targetOffset: Int): Unit = {
    val buffer = ByteBuffer.wrap(src, offset, length)
    var i = 0
    while (i < length / 4) {
      target(targetOffset + i) = buffer.getFloat()
      i += 1
    }
  }
}

object RoiRecordToFeature {
  def apply(convertLabel: Boolean = false, outKey: String = ImageFeature.bytes): RoiRecordToFeature
  = new RoiRecordToFeature(convertLabel, outKey)
}
