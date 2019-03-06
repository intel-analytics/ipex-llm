/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.inference

import java.io.{File, FileOutputStream, InputStream}
import java.nio.channels.Channels

import com.google.common.io.Files
import com.intel.analytics.zoo.pipeline.inference.DeviceType.DeviceTypeEnumVal
import com.intel.analytics.zoo.core.openvino.OpenvinoNativeLoader
import org.slf4j.LoggerFactory

import scala.language.postfixOps
import sys.process._

class OpenVinoInferenceSupportive {

  @native def loadOpenVinoIR(modelFilePath: String,
                             weightFilePath: String,
                             deviceTypeValue: Int): Long

  @native def predict(executableNetworkReference: Long,
                      data: Array[Float],
                      shape: Array[Int]): JTensor

  @native def releaseOpenVINOIR(executableNetworkReference: Long): Unit
}

object OpenVinoInferenceSupportive extends InferenceSupportive {
  val logger = LoggerFactory.getLogger(getClass)

  timing("load native so for openvino") {
    OpenvinoNativeLoader.load()
  }

  var openvinoTempDirDirPath: String = _
  timing("prepare openvino scripts") {
    val ovtmpDir = Files.createTempDir()
    openvinoTempDirDirPath = ovtmpDir.getCanonicalPath
    val OpenvinoNativeLoaderClass = (new OpenvinoNativeLoader()).getClass
    val zooShPath = "/zoo-optimize-model.sh"
    val zooShInputStream = OpenvinoNativeLoaderClass.getResourceAsStream(zooShPath)
    val zooShFile = writeFile(zooShInputStream, openvinoTempDirDirPath, zooShPath)
    val zooTarPath = "/model-optimizer.tar.gz"
    val zooTarInputStream = OpenvinoNativeLoaderClass.getResourceAsStream(zooTarPath)
    val zooTarFile = writeFile(zooTarInputStream, openvinoTempDirDirPath, zooTarPath)
    val zooConfPath = "/pipeline-configs.tar.gz"
    val zooConfInputStream = OpenvinoNativeLoaderClass.getResourceAsStream(zooConfPath)
    val zooConfFile = writeFile(zooConfInputStream, openvinoTempDirDirPath, zooConfPath)
    s"tar -xzvf ${zooTarFile.getAbsolutePath} -C $openvinoTempDirDirPath" !;
    s"tar -xzvf ${zooConfFile.getAbsolutePath} -C $openvinoTempDirDirPath" !;
    s"ls -alh $openvinoTempDirDirPath" !;
  }

  def loadTensorflowModel(modelPath: String,
                          modelType: String,
                          pipelineConfigPath: String,
                          extensionsConfigPath: String,
                          deviceType: DeviceTypeEnumVal): OpenVINOModel = {
    logger.info(s"start to optimize tensorflow model from " +
      s"$modelPath, $modelType, $pipelineConfigPath, $extensionsConfigPath")

    val modelName = modelPath.split("\\/").last.split("\\.").head

    modelType match {
      case null | "" => require(pipelineConfigPath != null
        && pipelineConfigPath != ""
        && extensionsConfigPath != null
        && extensionsConfigPath != "",
        s"modeltype is not provided, extensionsConfigPath, " +
          s"extensionsConfigPath should be specified")
      case _ => ModelType.isSupported(modelType) match {
          case true => logger.info(s"$modelType is supported." )
          case false => logger.warn(s"$modelType not supported, " +
            s"supported modeltypes are listed: ${ModelType.object_detection_types}")
        }
    }

    val actualPipelineConfigPath = pipelineConfigPath match {
      case null | "" =>
        val path = ModelType.resolveActualPipelineConfigPath(modelType)
        s"$openvinoTempDirDirPath/$path"
      case _ => pipelineConfigPath
    }
    val actualExtensionsConfigPath = extensionsConfigPath match {
      case null | "" =>
        val path = ModelType.resolveActualExtensionsConfigPath(modelType)
        s"$openvinoTempDirDirPath/$path"
      case _ => extensionsConfigPath
    }

    timing("load tensorflow model to openvino IR") {
      val loadTensorflowModelScriptPath: String = s"$openvinoTempDirDirPath/zoo-optimize-model.sh"
      val motfpyFilePath: String = s"$openvinoTempDirDirPath/model-optimizer/mo_tf.py"
      val tmpDir = Files.createTempDir()
      val outputPath: String = tmpDir.getCanonicalPath
      val stdout = new StringBuilder
      val stderr = new StringBuilder
      val log = ProcessLogger(stdout append _ + "\n", stderr append _ + "\n")
      val exitCode = timing("optimize tensorflow model") {
        Seq("sh",
          loadTensorflowModelScriptPath,
          modelPath,
          actualPipelineConfigPath,
          actualExtensionsConfigPath,
          outputPath,
          motfpyFilePath) ! log
      }
      logger.info(s"tensorflow model optimized, log: \n" +
        s"stderr: $stderr \n" +
        s"stdout: $stdout \n" +
        s"exitCode: $exitCode\n -----")
      exitCode match {
        case 0 => logger.info(s"tensorflow model optimization succeeded")
        case _ =>
          val message = stderr.toString().split("\n").filter(_ contains("ERROR")).mkString(",")
          throw
            new InferenceRuntimeException(s"Openvino optimize tf model error: $exitCode, $message")
      }
      val modelFilePath: String = s"$outputPath/$modelName.xml"
      val weightFilePath: String = s"$outputPath/$modelName.bin"
      val mappingFilePath: String = s"$outputPath/$modelName.mapping"
      val modelFile = new File(modelFilePath)
      val weightFile = new File(weightFilePath)
      val mappingFile = new File(mappingFilePath)
      val model = (modelFile.exists(), weightFile.exists(), mappingFile.exists()) match {
        case (true, true, true) =>
          loadOpenVinoIR(modelFilePath, weightFilePath, deviceType)
        case (_, _, _) => throw
          new InferenceRuntimeException("Openvino optimize tf model error")
      }
      timing("delete temporary model files") {
        modelFile.delete()
        weightFile.delete()
        mappingFile.delete()
      }
      model
    }
  }

  def loadOpenVinoIR(modelFilePath: String,
                     weightFilePath: String,
                     deviceType: DeviceTypeEnumVal): OpenVINOModel = {
    timing("load openvino IR") {
      val supportive: OpenVinoInferenceSupportive = new OpenVinoInferenceSupportive()
      val executableNetworkReference: Long =
        supportive.loadOpenVinoIR(modelFilePath, weightFilePath, deviceType.value)
      new OpenVINOModel(executableNetworkReference, supportive)
    }
  }

  def load(path: String): Unit = {
    logger.info(s"start to load library: $path.")
    val inputStream = OpenVinoInferenceSupportive.getClass.getResourceAsStream(s"/${path}")
    val file = File.createTempFile("OpenVinoInferenceSupportiveLoader", path)
    val src = Channels.newChannel(inputStream)
    val dest = new FileOutputStream(file).getChannel
    dest.transferFrom(src, 0, Long.MaxValue)
    dest.close()
    src.close()
    val filePath = file.getAbsolutePath
    logger.info(s"loading library: $path from $filePath ...")
    try {
      System.load(filePath)
    } finally {
      file.delete()
    }
  }

  def writeFile(inputStream: InputStream, dirPath: String, filePath: String): File = {
    val file = new File(s"$dirPath/$filePath")
    val src = Channels.newChannel(inputStream)
    val dest = new FileOutputStream(file).getChannel
    dest.transferFrom(src, 0, Long.MaxValue)
    dest.close()
    src.close()
    logger.info(s"file output to ${file.getAbsoluteFile} ...")
    file
  }
}

object ModelType {
  val logger = LoggerFactory.getLogger(getClass)
  val object_detection_types = List(
    "embedded_ssd_mobilenet_v1_coco",
    "facessd_mobilenet_v2_quantized_320x320_open_image_v4",
    "faster_rcnn_inception_resnet_v2_atrous_coco",
    "faster_rcnn_inception_resnet_v2_atrous_cosine_lr_coco",
    "faster_rcnn_inception_resnet_v2_atrous_oid",
    "faster_rcnn_inception_resnet_v2_atrous_pets",
    "faster_rcnn_inception_v2_coco",
    "faster_rcnn_inception_v2_pets",
    "faster_rcnn_nas_coco",
    "faster_rcnn_resnet101_atrous_coco",
    "faster_rcnn_resnet101_ava_v2.1",
    "faster_rcnn_resnet101_coco",
    "faster_rcnn_resnet101_fgvc",
    "faster_rcnn_resnet101_kitti",
    "faster_rcnn_resnet101_pets",
    "faster_rcnn_resnet101_voc07",
    "faster_rcnn_resnet152_coco",
    "faster_rcnn_resnet152_pets",
    "faster_rcnn_resnet50_coco",
    "faster_rcnn_resnet50_fgvc",
    "faster_rcnn_resnet50_pets",
    "mask_rcnn_inception_resnet_v2_atrous_coco",
    "mask_rcnn_inception_v2_coco",
    "mask_rcnn_resnet101_atrous_coco",
    "mask_rcnn_resnet101_pets",
    "mask_rcnn_resnet50_atrous_coco",
    "rfcn_resnet101_coco",
    "rfcn_resnet101_pets",
    "ssd_inception_v2_coco",
    "ssd_inception_v2_pets",
    "ssd_inception_v3_pets",
    "ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync",
    "ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync",
    "ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync",
    "ssd_mobilenet_v1_300x300_coco14_sync",
    "ssd_mobilenet_v1_coco",
    "ssd_mobilenet_v1_focal_loss_pets",
    "ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync",
    "ssd_mobilenet_v1_pets",
    "ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync",
    "ssd_mobilenet_v1_quantized_300x300_coco14_sync",
    "ssd_mobilenet_v2_coco",
    "ssd_mobilenet_v2_quantized_300x300_coco",
    "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync",
    "ssdlite_mobilenet_v1_coco",
    "ssdlite_mobilenet_v2_coco"
  )

  def isSupported(modelType : String): Boolean = {
    object_detection_types.contains(modelType)
  }

  def resolveActualPipelineConfigPath(modelType : String): String = {
   s"pipeline-configs/object_detection/$modelType.config"
  }

  def resolveActualExtensionsConfigPath(modelType : String): String = {
    val category = modelType match {
      case t if t.contains("ssd") => "ssd_v2"
      case t if t.contains("faster_rcnn") => "faster_rcnn"
      case t if t.contains("mask_rcnn") => "mask_rcnn"
      case t if t.contains("rfcn") => "rfcn"
    }
    s"model-optimizer/extensions/front/tf/${category}_support.json"
  }

}
