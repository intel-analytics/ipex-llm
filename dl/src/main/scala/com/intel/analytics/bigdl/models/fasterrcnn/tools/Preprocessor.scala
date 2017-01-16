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

package com.intel.analytics.bigdl.models.fasterrcnn.tools

import java.io.File
import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.{DataSet, LocalDataSet}
import com.intel.analytics.bigdl.models.fasterrcnn.dataset._
import com.intel.analytics.bigdl.models.fasterrcnn.dataset.transformers._
import com.intel.analytics.bigdl.models.fasterrcnn.model.{FasterRcnnParam, Phase}
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


object Preprocessor {
  private val PIXEL_MEANS_RGB = (122.7717f, 115.9465f, 102.9801f)

  def processImageSetsLocal(nodeNumber: Int, coreNumber: Int,
    imageSet: String, devkitPath: String, isTrain: Boolean = false,
    param: FasterRcnnParam): LocalDataSet[MiniBatch] = {
    val imageWithRois = loadImages(imageSet, devkitPath, isTrain, param)
    val transformers = ImgNormalizer(PIXEL_MEANS_RGB) -> ImageToMiniBatch(1, isTrain)
    (DataSet.array(imageWithRois) -> transformers).toLocal()
  }

  def processImageSetsDistri(nodeNumber: Int, coreNumber: Int,
    imageSet: String, devkitPath: String, isTrain: Boolean = false,
    param: FasterRcnnParam, sc: SparkContext): RDD[MiniBatch] = {
    val imageWithRois = loadImages(imageSet, devkitPath, isTrain, param)
    val transformers = ImgNormalizer(PIXEL_MEANS_RGB) -> ImageToMiniBatch(1, isTrain)
    sc.parallelize(imageWithRois, nodeNumber * coreNumber).mapPartitions(transformers(_))
  }

  private def loadImages(imageSet: String, devkitPath: String, isTrain: Boolean = false,
    param: FasterRcnnParam): Array[ImageWithRoi] = {
    val roiDataset = ObjectDataSource(imageSet, devkitPath, useFlipped = false)
    val imgReader = new LocalResizeImgReader(param.SCALES, param.SCALE_MULTIPLE_OF,
      param.phase == Phase.TRAIN)
    roiDataset.roidbs
      .map(roidb => imgReader.transform(roidb))
  }

  private def loadImages(folder: String, param: FasterRcnnParam): Array[ImageWithRoi] = {
    val imgReader = new LocalResizeImgReader(param.SCALES, param.SCALE_MULTIPLE_OF,
      param.phase == Phase.TRAIN)
    if (!new File(folder).exists()) {
      throw new Exception(s"cannot find image folder $folder")
    }
    new File(folder).listFiles().map(f => Roidb(f.getAbsolutePath))
      .map(roidb => imgReader.transform(roidb))
  }

  def processFolderLocal(folder: String, param: FasterRcnnParam, sc: SparkContext)
  : LocalDataSet[MiniBatch] = {
    val transformers = ImgNormalizer(PIXEL_MEANS_RGB) -> ImageToMiniBatch(1, isTrain = false)
    val imageWithRois = loadImages(folder, param)
    (DataSet.array(imageWithRois) -> transformers).toLocal()
  }

  def processFolderDisti(
    nodeNumber: Int, coreNumber: Int,
    folder: String,
    param: FasterRcnnParam, sc: SparkContext)
  : RDD[MiniBatch] = {
    val transformers = ImgNormalizer(PIXEL_MEANS_RGB) -> ImageToMiniBatch(1, isTrain = false)
    val imageWithRois = loadImages(folder, param)
    sc.parallelize(imageWithRois, nodeNumber * coreNumber)
      .mapPartitions(transformers(_))
  }

  def processSeqFileLocal(seqFloder: String, param: FasterRcnnParam,
    isTrain: Boolean = false): LocalDataSet[MiniBatch] = {
    val fileName = new File(seqFloder).listFiles().find(x =>
      x.getName.endsWith(".seq")).get.getName
    val total = fileName.substring(0, fileName.indexOf("-")).toInt
    (DataSet.SeqFileFolder.paths(Paths.get(seqFloder), total) ->
      LocalSeqToByteImageWithRoi() ->
      ImageResizer(param.SCALES, param.SCALE_MULTIPLE_OF, isTrain) ->
      ImgNormalizer(PIXEL_MEANS_RGB) -> ImageToMiniBatch(1, isTrain)).toLocal()
  }

  def processSeqFileDistri(
    nodeNumber: Int, coreNumber: Int,
    seqFloder: String,
    param: FasterRcnnParam,
    sc: SparkContext,
    isTrain: Boolean = false): RDD[MiniBatch] = {
    val transformers = ImageResizer(param.SCALES, param.SCALE_MULTIPLE_OF, isTrain) ->
      ImgNormalizer(PIXEL_MEANS_RGB) ->
      ImageToMiniBatch(1, isTrain)
    val recordToByteImageWithRoi = RecordToByteImageWithRoi()
    sc.sequenceFile(seqFloder, classOf[Text], classOf[Text])
      .map(x => recordToByteImageWithRoi.transform(x._1, x._2))
      .repartition(nodeNumber * coreNumber * 4)
      .mapPartitions(transformers(_))
  }
}
