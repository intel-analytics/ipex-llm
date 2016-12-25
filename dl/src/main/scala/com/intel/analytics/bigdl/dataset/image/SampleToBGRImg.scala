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

import com.intel.analytics.bigdl.dataset.{ByteRecord, Transformer}

import scala.collection.Iterator

object SampleToBGRImg {
  def apply(normalize: Float = 255f): SampleToBGRImg =
    new SampleToBGRImg(normalize)
}

class SampleToBGRImg(normalize: Float)
  extends Transformer[ByteRecord, LabeledBGRImage] {
  private val buffer = new LabeledBGRImage()

  override def apply(prev: Iterator[ByteRecord]): Iterator[LabeledBGRImage] = {
    prev.map(rawData => {
      buffer.copy(rawData.data, normalize).setLabel(rawData.label)
    })
  }
}
