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

package com.intel.analytics.bigdl.dataset.image

import com.intel.analytics.bigdl.dataset.{ByteRecord, Transformer}

import scala.collection.Iterator

object BytesToGreyImg {
  def apply(row: Int, col: Int): BytesToGreyImg
  = new BytesToGreyImg(row, col)
}

/**
 * Convert byte records into grey image.
 * @param row
 * @param col
 */
class BytesToGreyImg(row: Int, col: Int)
  extends Transformer[ByteRecord, LabeledGreyImage] {
  private val buffer = new LabeledGreyImage(row, col)

  override def apply(prev: Iterator[ByteRecord]): Iterator[LabeledGreyImage] = {
    prev.map(rawData => {
      require(row * col == rawData.data.length)
      buffer.setLabel(rawData.label).copy(rawData.data, 255.0f)
    })
  }
}
