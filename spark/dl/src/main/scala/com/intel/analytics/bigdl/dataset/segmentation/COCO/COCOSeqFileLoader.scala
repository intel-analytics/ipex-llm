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

package com.intel.analytics.bigdl.dataset.segmentation.COCO

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame}
import com.intel.analytics.bigdl.utils.{Engine, T}
import java.awt.image.DataBufferByte
import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import javax.imageio.ImageIO
import org.apache.hadoop.io.{ByteWritable, BytesWritable, Text}
import org.apache.spark.SparkContext

object COCOSeqFileLoader {
  val MAGIC_NUM = 0x1f3d4e5a
  /**
   * Extract hadoop sequence files from an HDFS path as ImageFrame
   * @param url sequence files folder path
   * @param sc spark context
   * @param partitionNum partition number, default: Engine.nodeNumber() * Engine.coreNumber()
   * @return
   */
  def filesToImageFrame(url: String, sc: SparkContext,
    partitionNum: Option[Int] = None): ImageFrame = {
    val num = partitionNum.getOrElse(Engine.nodeNumber() * Engine.coreNumber())
    val rawData = sc.sequenceFile(url, classOf[BytesWritable], classOf[BytesWritable], num)
      .map { data =>
        val metaBytes = new COCODeserializer(ByteBuffer.wrap(data._1.getBytes))
        val fileName = metaBytes.getString
        val (height, width, anno) = metaBytes.getAnnotations

        val labelClasses = Tensor(anno.map(_.categoryId.toFloat), Array(anno.length))
        val bboxes = Tensor(
          anno.toIterator.flatMap(ann => {
            Iterator(ann.bbox1, ann.bbox2, ann.bbox3, ann.bbox4)
          }).toArray,
          Array(anno.length, 4))
        val isCrowd = Tensor(anno.map(ann => if (ann.isCrowd) 1f else 0f), Array(anno.length))
        val masks = anno.map(ann => Tensor(ann.rleCounts, Array(ann.rleCounts.length)))
        require(metaBytes.getInt == MAGIC_NUM, "Corrupt metadata")

        val bis = new ByteArrayInputStream(data._2.getBytes)
        val image = ImageIO.read(bis)
        val rawdata = image.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData()
        val imf = ImageFeature(rawdata, RoiLabel(labelClasses, bboxes, masks), fileName)
        imf(ImageFeature.originalSize) = (height, width, 3)
        imf(RoiLabel.ISCROWD) = isCrowd
        imf
      }
    ImageFrame.rdd(rawData)
  }

  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName("Load COCO")
    val sc = new SparkContext(conf)
    Engine.init
    filesToImageFrame("/home/menooker/work/coco/seq2014", sc).toDistributed().rdd
      .foreach(imf => {
        println(imf(ImageFeature.uri))
      })
  }
}
