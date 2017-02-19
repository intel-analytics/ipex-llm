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

import java.io.FileInputStream
import java.nio.file.Paths
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.models.fasterrcnn.dataset.Roidb._
import com.intel.analytics.bigdl.models.fasterrcnn.utils.FileUtil
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{File => DlFile}

abstract class Imdb(val cacheFolder: String = "data/cache") {
  val name: String
  val classes: Array[String]
  var roidb: Array[Roidb] = _
  protected var imageIndex: Array[String] = _
  private var sizes: Array[Array[Int]] = _
  FileUtil.checkOrCreateDirs(cacheFolder)

  private def getWidth(i: Int) = {
    if (sizes == null) getImageSizes
    widths(i)
  }

  def widths: Array[Int] = {
    if (sizes == null) getImageSizes
    sizes(0)
  }

  def heights: Array[Int] = {
    if (sizes == null) getImageSizes
    sizes(1)
  }

  def getRoidb(useFlipped: Boolean): Array[Roidb] = {
    if (roidb != null && roidb.length > 0) return roidb
    roidb = loadRoidb
    if (useFlipped) {
      appendFlippedImages()
    }
    roidb
  }

  protected def loadRoidb: Array[Roidb]

  def numClasses: Int = classes.length

  def numImages: Int = imageIndex.length

  protected def imagePathAt(i: Int): String

  def imagePathFromIndex(index: String): String

  def appendFlippedImages(): Unit = {
    val newRoidb = new Array[Roidb](roidb.length * 2)
    roidb.copyToArray(newRoidb, 0)
    val isFlip = true
    var i = 0
    var r = 0
    while (i < numImages) {
      val roi = roidb(i)
      val boxes = roi.gtBoxes.clone()
      val oldx1 = boxes.select(2, 1)
      val oldx2 = boxes.select(2, 3)
      r = 1
      while (r <= boxes.size(1)) {
        boxes.setValue(r, 1, getWidth(i) - oldx2.valueAt(i + 1) - 1)
        boxes.setValue(r, 3, getWidth(i) - oldx1.valueAt(i + 1) - 1)
        r += 1
      }
      newRoidb(numImages + i) = Roidb(roi.imagePath, Target(roi.gtClasses, boxes),
        roi.gtOverlaps, isFlip)
      i += 1
    }
    val newImageIndex = new Array[String](imageIndex.length * 2)
    imageIndex.copyToArray(newImageIndex, 0)
    imageIndex.copyToArray(newImageIndex, imageIndex.length)
    imageIndex = newImageIndex
    roidb = newRoidb
  }

  private def getImageSizes: Array[Array[Int]] = {
    if (sizes != null) return sizes
    val cache_file = Paths.get(cacheFolder, name + "_image_sizes.pkl").toString
    if (FileUtil.existFile(cache_file)) {
      sizes = DlFile.load[Array[Array[Int]]](cache_file)
    } else {
      sizes = Array.ofDim[Int](2, numImages)
      var i = 0
      while (i < numImages) {
        val (width, height) = getImageSize(imagePathAt(i))
        sizes(0)(i) = width
        sizes(1)(i) = height
        i += 1
      }
      DlFile.save(sizes, cache_file)
    }
    sizes
  }

  def loadAnnotation(index: String): Roidb

  /**
   * load image width and height without loading the entire image
   * @param path image path
   * @return (width, height) tuple
   */
  def getImageSize(path: String): (Int, Int) = {
    var width = 0
    var height = 0
    val in = ImageIO.createImageInputStream(new FileInputStream(path))
    val readers = ImageIO.getImageReaders(in)
    if (readers.hasNext) {
      val reader = readers.next()
      reader.setInput(in)
      width = reader.getWidth(0)
      height = reader.getHeight(0)
      reader.dispose()
    }
    in.close()
    (width, height)
  }
}

case class Roidb(
  imagePath: String,
  target: Target = null,
  gtOverlaps: Tensor[Float] = null,
  flipped: Boolean = false) {
  var maxClasses = None: Option[Tensor[Float]]
  var maxOverlaps = None: Option[Tensor[Float]]
  var bboxTargets = None: Option[Tensor[Float]]

  def resizeGtRois(img: ImageWithRoi): Option[Target] = {
    resizeRois(img, target)
  }

  def gtBoxes: Tensor[Float] = target.bboxes
  def gtClasses: Tensor[Float] = target.classes
}

object Roidb {
  def resizeRois(img: ImageWithRoi, target: Target): Option[Target] = {
    var resizedTarget: Option[Target] = None
    if (resizedTarget != null) {
      val gtInds = target.classes.storage().array().zip(Stream from 1)
        .filter(x => x._1 != 0).map(x => x._2)
      val resizedBoxes = Tensor[Float](gtInds.length, 5)
      var i = 0
      while (i < gtInds.length) {
        resizedBoxes.setValue(i + 1, 1, target.bboxes.valueAt(gtInds(i), 1) * img.scaledH)
        resizedBoxes.setValue(i + 1, 2, target.bboxes.valueAt(gtInds(i), 2) * img.scaledW)
        resizedBoxes.setValue(i + 1, 3, target.bboxes.valueAt(gtInds(i), 3) * img.scaledH)
        resizedBoxes.setValue(i + 1, 4, target.bboxes.valueAt(gtInds(i), 4) * img.scaledW)
        resizedBoxes.setValue(i + 1, 5, target.classes.valueAt(gtInds(i)))
        i += 1
      }
      resizedTarget = Some(Target(null, resizedBoxes))
    }
    resizedTarget
  }
}

case class ByteImageWithRoi(data: Array[Byte], var dataLength: Int,
  var target: Target, path: String) {

  def resizeGtRois(img: ImageWithRoi): Option[Target] = {
    resizeRois(img, target)
  }

  def gtBoxes: Tensor[Float] = target.bboxes
  def gtClasses: Tensor[Float] = target.classes
}

class ImageWithRoi(
  var data: Array[Float],
  val imInfo: Tensor[Float], // (scaledHeight, scaledWidth, scaledh, scaledw)
  var target: Option[Target] = None) extends Serializable {

  var path: String = ""

  def this() = this(new Array[Float](0), Tensor[Float](4))

  def height: Int = imInfo.valueAt(1).toInt

  def width: Int = imInfo.valueAt(2).toInt

  def scaledH: Float = imInfo.valueAt(3)

  def scaledW: Float = imInfo.valueAt(4)

  def copy(rawData: Array[Byte]): this.type = {
    require(rawData.length == 8 + height * width * 3)
    if (data.length < height * width * 3) {
      data = new Array[Float](height * width * 3)
    }
    var i = 0
    while (i < width * height * 3) {
      data(i) = rawData(i + 8) & 0xff
      i += 1
    }
    this
  }

  def copyTo(storage: Array[Float], offset: Int, toRGB: Boolean = true): Unit = {
    val frameLength = width * height
    require(frameLength * 3 + offset <= storage.length)
    var j = 0
    if (toRGB) {
      while (j < frameLength) {
        storage(offset + j) = data(j * 3 + 2)
        storage(offset + j + frameLength) = data(j * 3 + 1)
        storage(offset + j + frameLength * 2) = data(j * 3)
        j += 1
      }
    } else {
      while (j < frameLength) {
        storage(offset + j) = data(j * 3)
        storage(offset + j + frameLength) = data(j * 3 + 1)
        storage(offset + j + frameLength * 2) = data(j * 3 + 2)
        j += 1
      }
    }
  }

}

object Imdb {
  /**
   * Get an imdb (image database) by name
   * @param name
   * @param devkitPath
   * @return
   */
  def getImdb(name: String, devkitPath: String): Imdb = {
    val items = name.split("_")
    if (items.length != 3) throw new Exception("dataset name error")
    if (items(0) == "voc") {
      new PascalVoc(items(1), items(2), devkitPath)
    } else {
      throw new UnsupportedOperationException("unsupported dataset")
    }
  }
}
