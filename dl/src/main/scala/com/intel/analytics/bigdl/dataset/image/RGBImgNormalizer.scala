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

import com.intel.analytics.bigdl.dataset.{LocalDataSet, Transformer}
import com.intel.analytics.bigdl.optim.LocalOptimizer._
import org.apache.log4j.Logger

import scala.collection.Iterator

object RGBImgNormalizer {
  val logger = Logger.getLogger(getClass)

  def apply(meanR: Double, meanG: Double, meanB: Double,
    stdR: Double, stdG: Double, stdB: Double): RGBImgNormalizer = {

    new RGBImgNormalizer(meanR, meanG, meanB, stdR, stdG, stdB)
  }

  def apply(mean: (Double, Double, Double), std: (Double, Double, Double)): RGBImgNormalizer = {
    new RGBImgNormalizer(mean._1, mean._2, mean._3, std._1, std._2, std._3)
  }

  def apply(dataSource: LocalDataSet[LabeledRGBImage])
  : RGBImgNormalizer = {
    apply(dataSource, -1)
  }


  def apply(dataSource: LocalDataSet[LabeledRGBImage], samples: Int)
  : RGBImgNormalizer = {
    var sumR: Double = 0
    var sumG: Double = 0
    var sumB: Double = 0
    var total: Long = 0
    dataSource.shuffle()
    var iter = dataSource.data(looped = false)
    val totalCount = if (samples < 0) dataSource.size() else samples
    var i = 1
    while (i <= totalCount) {
      val content = iter.next().content
      require(content.length % 3 == 0)
      var j = 0
      while (j < content.length) {
        sumR += content(j + 2)
        sumG += content(j + 1)
        sumB += content(j + 0)
        total += 1
        j += 3
      }
      i += 1
      logger.info(s"Mean: $i / $totalCount")
    }
    require(total > 0)
    val meanR = sumR / total
    val meanG = sumG / total
    val meanB = sumB / total
    sumR = 0
    sumG = 0
    sumB = 0
    i = 1
    iter = dataSource.data(looped = false)
    while (i <= totalCount) {
      val content = iter.next().content
      var j = 0
      while (j < content.length) {
        val diffR = content(j + 2) - meanR
        val diffG = content(j + 1) - meanG
        val diffB = content(j + 0) - meanB
        sumR += diffR * diffR
        sumG += diffG * diffG
        sumB += diffB * diffB
        j += 3
      }
      logger.info(s"Std: $i / $totalCount")
      i += 1
    }
    val stdR = math.sqrt(sumR / total)
    val stdG = math.sqrt(sumG / total)
    val stdB = math.sqrt(sumB / total)
    new RGBImgNormalizer(meanR, meanG, meanB, stdR, stdG, stdB)
  }
}

class RGBImgNormalizer(meanR: Double, meanG: Double, meanB: Double,
  stdR: Double, stdG: Double, stdB: Double)
  extends Transformer[LabeledRGBImage, LabeledRGBImage] {

  def getMean(): (Double, Double, Double) = (meanR, meanG, meanB)

  def getStd(): (Double, Double, Double) = (stdR, stdG, stdB)

  override def apply(prev: Iterator[LabeledRGBImage]): Iterator[LabeledRGBImage] = {
    prev.map(img => {
      val content = img.content
      require(content.length % 3 == 0)
      var i = 0
      while (i < content.length) {
        content(i + 2) = ((content(i + 2) - meanR) / stdR).toFloat
        content(i + 1) = ((content(i + 1) - meanG) / stdG).toFloat
        content(i + 0) = ((content(i + 0) - meanB) / stdB).toFloat
        i += 3
      }
      img
    })
  }
}
