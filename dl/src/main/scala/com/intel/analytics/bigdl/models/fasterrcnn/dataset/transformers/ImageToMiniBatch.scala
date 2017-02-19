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

package com.intel.analytics.bigdl.models.fasterrcnn.dataset.transformers

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.models.fasterrcnn.dataset.ImageWithRoi
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table

case class MiniBatch(data: Table, var label: Option[Table] = None,
  var path: String = "") {

  def this() = {
    this(new Table())
  }

  def getOrCreate(table: Table, ind: Int): Tensor[Float] = {
    if (table.length() < ind) {
      val t = Tensor[Float]()
      table.insert(t)
      t
    } else {
      table[Tensor[Float]](ind)
    }
  }

  def get(table: Table, ind: Int): Tensor[Float] = {
    if (table.length() < ind) {
      null
    } else {
      table[Tensor[Float]](ind)
    }
  }

  def getFeatures: Tensor[Float] = {
    getOrCreate(data, 1)
  }

  def getImInfo: Tensor[Float] = {
    getOrCreate(data, 2)
  }

  def height: Int = getImInfo.valueAt(1).toInt

  def width: Int = getImInfo.valueAt(2).toInt

  def scaledH: Float = getImInfo.valueAt(3)

  def scaledW: Float = getImInfo.valueAt(4)

  def getGtBoxesAndClasses: Tensor[Float] = {
    get(data, 3)
  }

  def getGtClasses: Tensor[Float] = {
    get(label.get, 1)
  }

  def getGtBoxes: Tensor[Float] = {
    get(label.get, 2)
  }

  private def resizeAndCopy(other: Tensor[Float], dest: Tensor[Float]): Unit = {
    if (other != null) dest.resizeAs(other).copy(other)
  }

  private def copyTable(other: Table, dest: Table): Unit = {
    var i = 1
    while (i <= other.length()) {
      resizeAndCopy(get(other, i), getOrCreate(dest, i))
      i += 1
    }
  }

  def copy(other: MiniBatch): MiniBatch = {
    copyTable(other.data, data)
    label = other.label
    path = other.path
    this
  }
}

class ImageToMiniBatch(batchSize: Int = 1, isTrain: Boolean)
  extends Transformer[ImageWithRoi, MiniBatch] {
  require(batchSize == 1)

  private var imgToTensor: ImageToTensor = null

  def apply(imgWithRoi: ImageWithRoi): MiniBatch = {
    if (imgToTensor == null) imgToTensor = new ImageToTensor()
    val data = new Table
    data.insert(imgToTensor(imgWithRoi))
    data.insert(imgWithRoi.imInfo)
    if (isTrain) {
      require(imgWithRoi.target.isDefined)
      data.insert(imgWithRoi.target.get.bboxes)
    }
    if (imgWithRoi.target.isDefined) {
      val label = imgWithRoi.target.get.toTable
      MiniBatch(data, Some(label), imgWithRoi.path)
    } else {
      MiniBatch(data, path = imgWithRoi.path)
    }
  }

  override def apply(prev: Iterator[ImageWithRoi]): Iterator[MiniBatch] = {
    new Iterator[MiniBatch] {
      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch = {
        if (prev.hasNext) {
          val imgWithRoi = prev.next()
          apply(imgWithRoi)
        } else {
          null
        }
      }
    }
  }
}

object ImageToMiniBatch {
  def apply(batchSize: Int = 1, isTrain: Boolean): ImageToMiniBatch
  = new ImageToMiniBatch(batchSize, isTrain)
}

class ImageToTensor(batchSize: Int = 1) extends Transformer[ImageWithRoi, Tensor[Float]] {
  require(batchSize == 1)

  private var featureTensor: Tensor[Float] = null
  private var featureData: Array[Float] = null

  def apply(imgWithRoi: ImageWithRoi): Tensor[Float] = {
    if (featureTensor == null) {
      featureTensor = Tensor[Float]()
    }
    if (featureData == null ||
      featureData.length < batchSize * imgWithRoi.height * imgWithRoi.width * 3) {
      featureData = new Array[Float](batchSize * imgWithRoi.height * imgWithRoi.width * 3)
    }
    imgWithRoi.copyTo(featureData, 0, toRGB = false)
    featureTensor.set(Storage[Float](featureData),
      storageOffset = 1, sizes = Array(batchSize, 3, imgWithRoi.height, imgWithRoi.width))
  }

  override def apply(prev: Iterator[ImageWithRoi]): Iterator[Tensor[Float]] = {
    new Iterator[Tensor[Float]] {


      override def hasNext: Boolean = prev.hasNext

      override def next(): Tensor[Float] = {
        if (prev.hasNext) {
          val imgWithRoi = prev.next()
          apply(imgWithRoi)
        } else {
          null
        }
      }
    }
  }
}

object ImageToTensor {
  def apply(img: ImageWithRoi): ImageToTensor = new ImageToTensor
}
