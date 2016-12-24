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

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.{DataSet, LocalDataSet, Transformer}

import scala.collection.Iterator

object GreyImgNormalizer {
  def apply(dataSource: DataSet[LabeledGreyImage], samples: Int = Int.MaxValue)
  : GreyImgNormalizer = {
    var sum: Double = 0
    var total: Int = 0
    dataSource.shuffle()
    var iter = dataSource.toLocal().data(looped = false)
    var i = 0
    while (i < math.min(samples, dataSource.size())) {
      val img = iter.next()
      img.content.foreach(e => {
        sum += e
        total += 1
      })
      i += 1
    }

    val mean = sum / total

    sum = 0
    i = 0
    iter = dataSource.toLocal().data(looped = false)
    while (i < math.min(samples, dataSource.size())) {
      val img = iter.next()
      img.content.foreach(e => {
        val diff = e - mean
        sum += diff * diff
      })
      i += 1
    }
    val std = math.sqrt(sum / total).toFloat
    new GreyImgNormalizer(mean, std)
  }

  def apply(mean : Double, std : Double): GreyImgNormalizer = {
    new GreyImgNormalizer(mean, std)
  }
}

class GreyImgNormalizer(mean : Double, std : Double)
  extends Transformer[LabeledGreyImage, LabeledGreyImage] {

  def getMean(): Double = mean

  def getStd(): Double = std

  override def apply(prev: Iterator[LabeledGreyImage]): Iterator[LabeledGreyImage] = {
    prev.map(img => {
      var i = 0
      val content = img.content
      while (i < content.length) {
        content(i) = ((content(i) - mean) / std).toFloat
        i += 1
      }
      img
    })
  }
}
