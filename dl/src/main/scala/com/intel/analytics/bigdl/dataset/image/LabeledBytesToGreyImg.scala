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

package com.intel.analytics.bigdl.dataset.image

import com.intel.analytics.bigdl.dataset.Transformer

import scala.collection.Iterator

object LabeledBytesToGreyImg {
  def apply(row: Int, col: Int): LabeledBytesToGreyImg
  = new LabeledBytesToGreyImg(row, col)
}

class LabeledBytesToGreyImg(row: Int, col: Int)
  extends Transformer[(Float, Array[Byte]), LabeledGreyImage] {
  private val buffer = new LabeledGreyImage(row, col)

  override def apply(prev: Iterator[(Float, Array[Byte])]): Iterator[LabeledGreyImage] = {
    prev.map(rawData => {
      require(row * col == rawData._2.length)
      require(rawData._1 >= 1)
      buffer.setLabel(rawData._1).copy(rawData._2, 255.0f)
    })
  }
}
