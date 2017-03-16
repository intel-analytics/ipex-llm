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

package com.intel.analytics.bigdl.models.fasterrcnn.dataset

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.roiimage.RoiImage
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table

/**
 * the input of model
 * @param data if in test phase: (imgData, imInfo)
 *             if in train phase: (imgData, imInfo, GtBoxesAndClasses)
 * @param label
 * @param path image path
 */
case class Sample(data: Table, var label: Option[Table] = None,
  var path: String = "") {

  def this() = {
    this(new Table())
  }

  def height: Int = data[Tensor[Float]](2).valueAt(1).toInt

  def width: Int = data[Tensor[Float]](2).valueAt(2).toInt

  /**
   * @return ground truth classes
   */
  def getGtClasses: Tensor[Float] = {
    if (label.isDefined) label.get[Tensor[Float]](1)
    else null
  }

  /**
   * @return ground truth boxes
   */
  def getGtBoxes: Tensor[Float] = {
    if (label.isDefined) label.get[Tensor[Float]](2)
    else null
  }

  private def getOrCreate(table: Table, ind: Int): Tensor[Float] = {
    if (table.length() < ind) {
      val t = Tensor[Float]()
      table.insert(t)
      t
    } else {
      table[Tensor[Float]](ind)
    }
  }

  private def getTensor(table: Table, ind: Int): Tensor[Float] = {
    if (table.length() < ind) {
      null
    } else {
      table[Tensor[Float]](ind)
    }
  }

  private def resizeAndCopy(other: Tensor[Float], dest: Tensor[Float]): Unit = {
    if (other != null) dest.resizeAs(other).copy(other)
  }

  private def copyTable(other: Table, dest: Table): Unit = {
    var i = 1
    while (i <= other.length()) {
      resizeAndCopy(getTensor(other, i), getOrCreate(dest, i))
      i += 1
    }
  }

  def copy(other: Sample): Sample = {
    copyTable(other.data, data)
    label = other.label
    path = other.path
    this
  }
}

class ImageToSample(batchSize: Int = 1, isTrain: Boolean)
  extends Transformer[RoiImage, Sample] {
  require(batchSize == 1)

  private var imgToTensor: ImageToTensor = null

  def apply(imgWithRoi: RoiImage): Sample = {
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
      Sample(data, Some(label), imgWithRoi.path)
    } else {
      Sample(data, path = imgWithRoi.path)
    }
  }

  override def apply(prev: Iterator[RoiImage]): Iterator[Sample] = {
    new Iterator[Sample] {
      override def hasNext: Boolean = prev.hasNext

      override def next(): Sample = {
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

object ImageToSample {
  def apply(batchSize: Int = 1, isTrain: Boolean): ImageToSample
  = new ImageToSample(batchSize, isTrain)
}

class ImageToTensor(batchSize: Int = 1) extends Transformer[RoiImage, Tensor[Float]] {
  require(batchSize == 1)

  private var featureTensor: Tensor[Float] = _
  private var featureData: Array[Float] = _

  def apply(imgWithRoi: RoiImage): Tensor[Float] = {
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

  override def apply(prev: Iterator[RoiImage]): Iterator[Tensor[Float]] = {
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
  def apply(img: RoiImage): ImageToTensor = new ImageToTensor
}
