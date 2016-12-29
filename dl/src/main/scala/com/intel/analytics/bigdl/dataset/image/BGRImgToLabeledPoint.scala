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
import org.apache.log4j.Logger
import org.apache.spark.mllib.linalg.DenseVector

import scala.collection.Iterator

object BGRImgToDfPoint {
  val logger = Logger.getLogger(getClass)

  def apply(): BGRImgToDfPoint = {
    new BGRImgToDfPoint()
  }
}

case class DfPoints(label: String, features: DenseVector)

class BGRImgToDfPoint(pathName : Array[LocalLabeledImagePath] = null)
  extends Transformer[LabeledBGRImage, DfPoints] {

  private var featureData: Array[Float] = null

  override def apply(prev: Iterator[LabeledBGRImage]): Iterator[DfPoints] = {
    prev.map(
      img => {
        if (null == featureData) {
          featureData = new Array[Float](3 * img.height() * img.width())
        }
        img.copyTo(featureData, 0, true)
        new DfPoints(img.label().toString, new DenseVector(featureData.map(_.toDouble)))
      }
    )
  }
}
