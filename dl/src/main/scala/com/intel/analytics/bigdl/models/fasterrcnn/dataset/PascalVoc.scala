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

import java.nio.file.Paths

import com.intel.analytics.bigdl.models.fasterrcnn.utils.FileUtil
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{File => DlFile}

import scala.Array._
import scala.io.Source
import scala.xml.XML

/**
 * Parse the pascal voc dataset, load images and annotations
 * @param year       the year of dataset
 * @param imageSet   train, val, test, etc
 * @param devkitPath dataset folder
 */
class PascalVoc(val year: String = "2007", val imageSet: String,
  devkitPath: String)
  extends Imdb {

  override val name = "voc_" + year + "_" + imageSet
  val dataPath = Paths.get(devkitPath, "/VOC" + year).toString

  override val classes = Array[String](
    "__background__", // always index 0
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
  )
  val classToInd = (classes zip (Stream from 1)).toMap
  imageIndex = loadImageSetIndex()

  val useDiff = false

  assert(FileUtil.existFile(devkitPath),
    "VOCdevkit path does not exist: " + devkitPath)
  assert(FileUtil.existFile(dataPath),
    "Path does not exist: {}" + dataPath)

  /**
   * Return the absolute path to image i in the image sequence.
   * @param i seq index
   * @return image path
   */
  def imagePathAt(i: Int): String = imagePathFromIndex(imageIndex(i))

  /**
   * Construct an image path from the image"s "index" identifier.
   * @param index e.g. 000001
   * @return image path
   */
  def imagePathFromIndex(index: String): String = s"$dataPath/JPEGImages/$index.jpg"


  /**
   * Load the indexes listed in this dataset's image set file.
   *
   * Example path to image set file: devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
   * @return
   */
  def loadImageSetIndex(): Array[String] = {
    val imageSetFile = Paths.get(dataPath, s"/ImageSets/Main/$imageSet.txt").toFile
    assert(imageSetFile.exists(), "Path does not exist " + imageSetFile.getAbsolutePath)
    Source.fromFile(imageSetFile).getLines().map(line => line.trim).toArray
  }

  /**
   * Load image and bounding boxes info from XML file in the PASCAL VOC
   * format.
   */
  def loadAnnotation(index: String): Roidb = {
    val xml = XML.loadFile(dataPath + "/Annotations/" + index + ".xml")
    var objs = xml \\ "object"

    if (!useDiff) {
      // Exclude the samples labeled as difficult
      objs = objs.filter(obj => (obj \ "difficult").text.toInt == 0)
    }

    val boxes = Tensor[Float](objs.length, 4)
    val gtClasses = Tensor[Float](objs.length)
    // Load object bounding boxes into a data frame.
    var ix = 1
    while (ix <= objs.length) {
      // pixel indexes 0-based
      val obj = objs(ix - 1)
      val bndbox = obj \ "bndbox"
      val x1 = (bndbox \ "xmin").text.toFloat - 1
      val y1 = (bndbox \ "ymin").text.toFloat - 1
      val x2 = (bndbox \ "xmax").text.toFloat - 1
      val y2 = (bndbox \ "ymax").text.toFloat - 1
      val cls = classToInd((obj \ "name").text)
      boxes.setValue(ix, 1, x1)
      boxes.setValue(ix, 2, y1)
      boxes.setValue(ix, 3, x2)
      boxes.setValue(ix, 4, y2)
      gtClasses.setValue(ix, cls)
      ix += 1
    }
    Roidb(imagePathFromIndex(index), Target(gtClasses, boxes), flipped = false)
  }

  /**
   * This function loads/saves from/to a cache file to speed up future calls.
   * @return the database of ground-truth regions of interest.
   */
  def loadRoidb: Array[Roidb] = {
    val cacheFile = Paths.get(cacheFolder, name + "_gt_roidb.pkl").toString
    val gtRoidb = FileUtil.load[Array[Roidb]](cacheFile)

    gtRoidb match {
      case Some(gt) => gt
      case _ =>
        roidb = imageIndex.map(index => loadAnnotation(index))
        DlFile.save(roidb, cacheFile, isOverwrite = true)
        roidb
    }
  }
}
