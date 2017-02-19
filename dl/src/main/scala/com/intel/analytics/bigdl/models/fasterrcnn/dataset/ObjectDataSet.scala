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

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.dataset.LocalDataSet
import com.intel.analytics.bigdl.dataset.image.BGRImage
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{RandomGenerator, Table}

import scala.util.Random


class ObjectDataSource(val imdb: Imdb, val useFlipped: Boolean = false)
  extends LocalDataSet[Roidb] {

  val roidbs = imdb.getRoidb(useFlipped)

  // permutation of the data index
  var perm: Array[Int] = _

  override def shuffle(): Unit = {
    // Make minibatches from images that have similar aspect ratios (i.e. both
    // tall and thin or both short and wide) in order to avoid wasting computation
    // on zero-padding.
    def shuffleWithAspectGrouping(widths: Array[Int], heights: Array[Int]): Unit = {
      val horz = (widths zip heights).map(x => x._1 >= x._2)
      val vert = horz.map(x => !x)
      val horzInds = horz.zipWithIndex.filter(x => x._1).map(x => x._2)
      val vertInds = vert.zipWithIndex.filter(x => x._1).map(x => x._2)
      val indsArr = (Random.shuffle(horzInds.toSeq) ++ Random.shuffle(vertInds.toSeq)).toArray
      val rowPerm = Random.shuffle(Seq.range(0, indsArr.length / 2))
      var i = 0
      while (i < rowPerm.length) {
        perm(rowPerm(i) * 2) = indsArr(i * 2)
        perm(rowPerm(i) * 2 + 1) = indsArr(i * 2 + 1)
        i += 1
      }
    }
    if (imdb.widths != null && imdb.widths.length > 0) {
      shuffleWithAspectGrouping(imdb.widths, imdb.heights)
    } else {
      RandomGenerator.shuffle(perm)
    }
  }

  /**
   * Get a sequence of data
   * @param looped
   * @return data sequence
   */
  override def data(looped: Boolean): Iterator[Roidb] = {
    perm = roidbs.indices.toArray
    new Iterator[Roidb] {
      private val index = new AtomicInteger()

      override def hasNext: Boolean = {
        if (looped) {
          true
        } else {
          index.get() < perm.length
        }
      }

      override def next(): Roidb = {
        val curIndex = index.getAndIncrement()
        if (looped || curIndex < perm.length) {
          roidbs(perm(if (looped) curIndex % perm.length else curIndex))
        } else {
          null.asInstanceOf[Roidb]
        }
      }
    }
  }

  override def size(): Long = roidbs.length
}

object ObjectDataSource {
  def apply(name: String, devkitPath: String, useFlipped: Boolean): ObjectDataSource =
    new ObjectDataSource(Imdb.getImdb(name, devkitPath), useFlipped)
}


object BGRImageOD {


  // Max pixel size of the longest side of a scaled input image
  val MAX_SIZE = 1000f

  /**
   *
   * @param img
   * @param scales
   * @param scaleMultipleOf Resize test images so that its width and height are multiples of ...
   * @return
   */
  def getWidthHeightAfterRatioScale(img: BufferedImage, scales: Array[Int],
    scaleMultipleOf: Int, imInfo: Tensor[Float]): Tensor[Float] = {
    val scaleTo = scales(Random.nextInt(scales.length))
    val imSizeMin = Math.min(img.getWidth, img.getHeight)
    val imSizeMax = Math.max(img.getWidth, img.getHeight)
    var im_scale = scaleTo.toFloat / imSizeMin.toFloat
    // Prevent the biggest axis from being more than MAX_SIZE
    if (Math.round(im_scale * imSizeMax) > MAX_SIZE) {
      im_scale = MAX_SIZE / imSizeMax.toFloat
    }

    val imScaleH = (Math.floor(img.getHeight * im_scale / scaleMultipleOf) *
      scaleMultipleOf / img.getHeight).toFloat
    val imScaleW = (Math.floor(img.getWidth * im_scale / scaleMultipleOf) *
      scaleMultipleOf / img.getWidth).toFloat
    imInfo.setValue(1, imScaleH * img.getHeight)
    imInfo.setValue(2, imScaleW * img.getWidth)
    imInfo.setValue(3, imScaleH)
    imInfo.setValue(4, imScaleW)
  }

  def readImage(path: Path, scales: Array[Int], scaleMultipleOf: Int, imInfo: Tensor[Float])
  : Array[Byte] = {
    val img: BufferedImage = BGRImage.readRawImage(path)
    if (img == null) throw new Exception(s"$path cannot be loaded correctly")
    getWidthHeightAfterRatioScale(img, scales, scaleMultipleOf, imInfo)
    BGRImage.resizeImage(img, imInfo.valueAt(2).toInt, imInfo.valueAt(1).toInt)
  }
}

case class Target(classes: Tensor[Float], bboxes: Tensor[Float]) {
  def toTable: Table = {
    val table = new Table()
    table.insert(classes)
    table.insert(bboxes)
  }
}
