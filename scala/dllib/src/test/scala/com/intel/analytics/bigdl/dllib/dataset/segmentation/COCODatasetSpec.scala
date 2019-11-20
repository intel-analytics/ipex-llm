/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.dataset.segmentation

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, RoiImageInfo}
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import java.awt.image.DataBufferByte
import java.io.{File, FileInputStream}
import javax.imageio.ImageIO
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class COCODatasetSpec extends FlatSpec with Matchers with BeforeAndAfter {

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  val resourcePath: String = processPath(getClass.getClassLoader.getResource("coco").getPath)
  val dataSet: COCODataset = COCODataset.load(resourcePath
      + File.separator + "cocomini.json", resourcePath)

  "COCODataset" should "correctly be loaded" in {
    dataSet.images.length should be (5)
    dataSet.annotations.length should be (6)
    val cateIdx = Array(53, 53, 53, 1, 19, 1).toIterator
    val sizes = Array((428, 640), (480, 640), (427, 640), (480, 640), (427, 640)).toIterator
    for (anno <- dataSet.annotations) {
      anno.image.id should be (anno.imageId)
      dataSet.categoryId2Idx(anno.categoryId) should be (cateIdx.next())
      anno.categoryIdx should be (dataSet.categoryId2Idx(anno.categoryId))
      if (anno.isCrowd) {
        anno.segmentation.isInstanceOf[COCORLE] should be (true)
      } else {
        anno.segmentation.isInstanceOf[COCOPoly] should be (true)
        val poly = anno.segmentation.asInstanceOf[COCOPoly]
        poly.height should be (anno.image.height)
        poly.width should be (anno.image.width)
      }
    }
    for (img <- dataSet.images) {
      val size = sizes.next()
      img.height should be (size._1)
      img.width should be (size._2)
    }
    for (i <- 1 to dataSet.categories.length) {
      val cate = dataSet.getCategoryByIdx(i)
      dataSet.categoryId2Idx(cate.id) should be (i)
    }
  }

  "COCODataset.toImageFeatures" should "correctly work" in {
    val cateIdx = Array(1, 19, 53, 53, 53, 1).toIterator
    val sizes = Array((428, 640, 3), (480, 640, 3), (427, 640, 3), (480, 640, 3),
      (427, 640, 3)).toIterator
    val uri = Array("COCO_val2014_000000153344.jpg", "COCO_val2014_000000091136.jpg",
      "COCO_val2014_000000558840.jpg", "COCO_val2014_000000200365.jpg",
      "COCO_val2014_000000374530.jpg"
    ).toIterator
    val isCrowd = Array(1f, 1f, 0f, 0f, 0f, 1f).toIterator
    dataSet.toImageFeatures.foreach(imf => {
      imf.getOriginalSize should be (sizes.next())
      val iscr = imf[Tensor[Float]](RoiImageInfo.ISCROWD)

      val roilabel = imf.getLabel[RoiLabel]
      roilabel.classes.size() should be (iscr.size())
      for(i <- 1 to iscr.nElement()) {
        iscr.valueAt(i) should be (isCrowd.next())
        roilabel.classes.valueAt(i) should be (cateIdx.next())
      }
      roilabel.bboxes.size() should be (Array(roilabel.classes.size(1), 4))

      val inputStream = new FileInputStream(resourcePath + File.separator + uri.next())
      val image = ImageIO.read(inputStream)
      val rawdata = image.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData()
      require(java.util.Arrays.equals(rawdata, imf[Array[Byte]](ImageFeature.bytes)))
    })
  }

  "COCOImage.toTable" should "correctly work" in {
    val cateIdx = Array(1, 19, 53, 53, 53, 1).toIterator
    val sizes = Array((428, 640, 3), (480, 640, 3), (427, 640, 3), (480, 640, 3),
      (427, 640, 3)).toIterator
    val isCrowd = Array(1f, 1f, 0f, 0f, 0f, 1f).toIterator
    dataSet.images.map(_.toTable).foreach(tab => {
      RoiImageInfo.getOrigSize(tab) should be (sizes.next())
      val iscr = RoiImageInfo.getIsCrowd(tab)
      val classes = RoiImageInfo.getClasses(tab)
      classes.size() should be (iscr.size())
      for(i <- 1 to iscr.nElement()) {
        iscr.valueAt(i) should be (isCrowd.next())
        classes.valueAt(i) should be (cateIdx.next())
      }
      RoiImageInfo.getBBoxes(tab).size() should be (Array(classes.size(1), 4))

    })
  }

}
