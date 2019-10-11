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

import java.io.{ByteArrayInputStream, File, FileOutputStream, InputStream}
import java.nio.channels.Channels

import com.intel.analytics.zoo.common.Utils
import com.intel.analytics.zoo.pipeline.inference.DeviceType.DeviceTypeEnumVal
import com.intel.analytics.zoo.core.openvino.OpenvinoNativeLoader
import org.slf4j.LoggerFactory

import scala.io.Source
import scala.language.postfixOps
import sys.process._

class OpenVinoInferenceSupportive extends InferenceSupportive with Serializable {

  @native def loadOpenVinoIR(modelFilePath: String,
                             weightFilePath: String,
                             deviceTypeValue: Int,
                             batchSize: Int): Long

  @native def loadOpenVinoIRInt8(modelFilePath: String,
                                 weightFilePath: String,
                                 deviceTypeValue: Int,
                                 batchSize: Int): Long

  @native def predict(executableNetworkReference: Long,
                      data: Array[Float],
                      shape: Array[Int]): JTensor

  @native def predictInt8(executableNetworkReference: Long,
                      data: Array[Float],
                      shape: Array[Int]): JTensor

  @native def predictInt8(executableNetworkReference: Long,
                      data: Array[Byte],
                      shape: Array[Int]): JTensor

  @native def releaseOpenVINOIR(executableNetworkReference: Long): Unit
}

object OpenVinoInferenceSupportive extends InferenceSupportive with Serializable {
  val logger = LoggerFactory.getLogger(getClass)

  timing("load native so for openvino") {
    OpenvinoNativeLoader.load()
  }

  val optimizeObjectDetectionRelativePath = "/zoo-optimize-model-tf-object-detection.sh"
  val optimizeImageClassificationRelativePath = "/zoo-optimize-model-tf-image-classification.sh"
  val optimizeImageClassificationFromSavedModelRelativePath
    = "/zoo-optimize-model-tf-image-classification-savedmodel.sh"
  val calibrateRelativePath = "/zoo-calibrate-model.sh"
  var openvinoTempDirPath: String = _
  var optimizeObjectDetectionSHPath: String = _
  var optimizeImageClassificationSHPath: String = _
  var optimizeImageClassificationFromSavedModelSHPath: String = _
  var motfpyFilePath: String = _
  var calibrateSHPath: String = _
  var calibrationToolPath: String = _
  var calibrationLibPath: String = _

  timing("prepare openvino scripts") {
    val ovtmpDir = Utils.createTmpDir("ZooVino").toFile()
    openvinoTempDirPath = ovtmpDir.getCanonicalPath
    optimizeObjectDetectionSHPath = s"$openvinoTempDirPath$optimizeObjectDetectionRelativePath"
    optimizeImageClassificationSHPath =
      s"$openvinoTempDirPath$optimizeImageClassificationRelativePath"
    optimizeImageClassificationFromSavedModelSHPath
      = s"$openvinoTempDirPath$optimizeImageClassificationFromSavedModelRelativePath"
    motfpyFilePath = s"$openvinoTempDirPath/model-optimizer/mo_tf.py"
    calibrateSHPath = s"$openvinoTempDirPath$calibrateRelativePath"
    calibrationToolPath = s"$openvinoTempDirPath/inference-engine-bin/calibration_tool"
    calibrationLibPath = s"$openvinoTempDirPath/inference-engine-bin/lib"

    val OpenvinoNativeLoaderClass = (new OpenvinoNativeLoader()).getClass

    val optimizeTFODPath = optimizeObjectDetectionRelativePath
    val optimizeTFODInputStream = OpenvinoNativeLoaderClass.getResourceAsStream(optimizeTFODPath)
    writeFile(optimizeTFODInputStream, openvinoTempDirPath, optimizeTFODPath)

    val optimizeTFICPath = optimizeImageClassificationRelativePath
    val optimizeTFICInputStream = OpenvinoNativeLoaderClass.getResourceAsStream(optimizeTFICPath)
    writeFile(optimizeTFICInputStream, openvinoTempDirPath, optimizeTFICPath)

    val optimizeTFIC4SavedModelPath = optimizeImageClassificationFromSavedModelRelativePath
    val optimizeTFIC4SavedModelInputStream
      = OpenvinoNativeLoaderClass.getResourceAsStream(optimizeTFIC4SavedModelPath)
    writeFile(optimizeTFIC4SavedModelInputStream, openvinoTempDirPath, optimizeTFIC4SavedModelPath)

    val calibrateTFPath = calibrateRelativePath
    val calibrateTFInputStream = OpenvinoNativeLoaderClass.getResourceAsStream(calibrateTFPath)
    writeFile(calibrateTFInputStream, openvinoTempDirPath, calibrateTFPath)

    val moTarPath = "/model-optimizer.tar.gz"

    val moTarInputStream = scala.util.Properties.isMac match {
      case true => OpenvinoNativeLoaderClass
        .getResourceAsStream("/darwin-x86_64" + moTarPath)
      case false => OpenvinoNativeLoaderClass
        .getResourceAsStream(moTarPath)
    }
    val moTarFile = writeFile(moTarInputStream, openvinoTempDirPath, moTarPath)
    s"tar -xzvf ${moTarFile.getAbsolutePath} -C $openvinoTempDirPath" !;

    val ieTarPath = "/inference-engine-bin.tar.gz"
    val ieTarInputStream = scala.util.Properties.isMac match {
      case true => OpenvinoNativeLoaderClass
        .getResourceAsStream("/darwin-x86_64" + ieTarPath)
      case false => OpenvinoNativeLoaderClass
        .getResourceAsStream(ieTarPath)
    }
    val ieTarFile = writeFile(ieTarInputStream, openvinoTempDirPath, ieTarPath)
    s"tar -xzvf ${ieTarFile.getAbsolutePath} -C $openvinoTempDirPath" !;

    val igTarPath = "/inference-graphs.tar.gz"
    val igTarInputStream = OpenvinoNativeLoaderClass.getResourceAsStream(igTarPath)
    val igTarFile = writeFile(igTarInputStream, openvinoTempDirPath, igTarPath)
    s"tar -xzvf ${igTarFile.getAbsolutePath} -C $openvinoTempDirPath" !;

    val pcTarPath = "/pipeline-configs.tar.gz"
    val pcTarInputStream = OpenvinoNativeLoaderClass.getResourceAsStream(pcTarPath)
    val pcTarFile = writeFile(pcTarInputStream, openvinoTempDirPath, pcTarPath)
    s"tar -xzvf ${pcTarFile.getAbsolutePath} -C $openvinoTempDirPath" !;

    s"ls -alh $openvinoTempDirPath" !;
  }

  def optimizeTFImageClassificationModel(modelPath: String,
                                     modelType: String,
                                     checkpointPath: String,
                                     inputShape: Array[Int],
                                     ifReverseInputChannels: Boolean,
                                     meanValues: Array[Float],
                                     scale: Float,
                                     outputDir: String): Unit = {
    logger.info(s"start to optimize tf image classification model from " +
      s"$modelPath, $modelType, $checkpointPath, $inputShape, " +
      s"$ifReverseInputChannels, $meanValues, $scale, to $outputDir")

    modelType match {
      case null | "" =>
        require(modelPath != null && modelPath != "",
          "modeltype is not provided, modelPath should be specified")
      case _ =>
        ModelType.isSupportedImageClassificationModel(modelType) match {
          case true => logger.info(s"$modelType is supported." )
          case false => logger.warn(s"$modelType not supported, " +
            s"supported tf image classification model types are listed: " +
            s"${ModelType.image_classification_types}")
        }
    }

    val actualModelPath = modelPath match {
      case null | "" =>
        val path = ModelType.resolveActualInferenceGraphPath(modelType)
        s"$openvinoTempDirPath/$path"
      case _ => modelPath
    }

    timing("optimize tf image classification model to openvino IR") {
      val outputPath: String = outputDir
      val inputShapeStr = inputShape.mkString("[", ",", "]")
      val ifReverseInputChannelsStr = ifReverseInputChannels match {
        case true => "1"
        case false => "0"
      }
      val meanValuesStr = meanValues.mkString("[", ",", "]")
      val scaleStr = scale + ""

      val stdout = new StringBuilder
      val stderr = new StringBuilder
      val log = ProcessLogger(stdout append _ + "\n", stderr append _ + "\n")
      val exitCode = timing("optimize tf image classification model execution") {
        Seq("sh",
          optimizeImageClassificationSHPath,
          actualModelPath,
          checkpointPath,
          inputShapeStr,
          ifReverseInputChannelsStr,
          meanValuesStr,
          scaleStr,
          outputPath,
          motfpyFilePath) ! log
      }
      logger.info(s"tf image classification model optimized, log: \n" +
        s"stderr: $stderr \n" +
        s"stdout: $stdout \n" +
        s"exitCode: $exitCode\n -----")
      exitCode match {
        case 0 => logger.info(s"tf image classification model optimization succeeded")
        case _ =>
          val message = stderr.toString().split("\n").filter(_ contains ("ERROR")).mkString(",")
          throw
            new InferenceRuntimeException(s"Openvino optimize tf image classification " +
              s"model error: " +
              s"$exitCode, $message")
      }
    }

  }

  def optimizeTFImageClassificationModel(savedModelDir: String,
                                         inputShape: Array[Int],
                                         ifReverseInputChannels: Boolean,
                                         meanValues: Array[Float],
                                         scale: Float,
                                         input: String,
                                         outputDir: String): Unit = {
    timing("optimize tf image classification model to openvino IR") {
      val outputPath: String = outputDir
      val inputShapeStr = inputShape.mkString("[", ",", "]")
      val ifReverseInputChannelsStr = ifReverseInputChannels match {
        case true => "1"
        case false => "0"
      }
      val meanValuesStr = meanValues.mkString("[", ",", "]")
      val scaleStr = scale + ""
      val inputStr = input

      val stdout = new StringBuilder
      val stderr = new StringBuilder
      val log = ProcessLogger(stdout append _ + "\n", stderr append _ + "\n")
      val exitCode = timing("optimize tf image classification model execution") {
        Seq("sh",
          optimizeImageClassificationFromSavedModelSHPath,
          savedModelDir,
          inputShapeStr,
          ifReverseInputChannelsStr,
          meanValuesStr,
          scaleStr,
          inputStr,
          outputPath,
          motfpyFilePath) ! log
      }
      logger.info(s"tf image classification model optimized, log: \n" +
        s"stderr: $stderr \n" +
        s"stdout: $stdout \n" +
        s"exitCode: $exitCode\n -----")
      exitCode match {
        case 0 => logger.info(s"tf image classification model optimization succeeded")
        case _ =>
          val message = stderr.toString().split("\n").filter(_ contains ("ERROR")).mkString(",")
          throw
            new InferenceRuntimeException(s"Openvino optimize tf image classification " +
              s"model error: " +
              s"$exitCode, $message")
      }
    }
  }

  def optimizeTFObjectDetectionModel(modelPath: String,
                                     modelType: String,
                                     pipelineConfigPath: String,
                                     extensionsConfigPath: String,
                                     outputDir: String): Unit = {
    logger.info(s"start to optimize tf object detection model from " +
      s"$modelPath, $modelType, $pipelineConfigPath, $extensionsConfigPath, to $outputDir")

    val modelName = modelPath.split("\\/").last.split("\\.").head

    modelType match {
      case null | "" =>
        require(pipelineConfigPath != null
          && pipelineConfigPath != ""
          && extensionsConfigPath != null
          && extensionsConfigPath != "",
          s"modeltype is not provided, extensionsConfigPath, " +
            s"extensionsConfigPath should be specified")
      case _ =>
        ModelType.isSupportedObjectDetectionModel(modelType) match {
          case true => logger.info(s"$modelType is supported." )
          case false => logger.warn(s"$modelType not supported, " +
            s"supported tf object detection model types are listed: " +
            s"${ModelType.object_detection_types}")
      }
    }

    val actualPipelineConfigPath = pipelineConfigPath match {
      case null | "" =>
        val path = ModelType.resolveActualPipelineConfigPath(modelType)
        s"$openvinoTempDirPath/$path"
      case _ => pipelineConfigPath
    }
    val actualExtensionsConfigPath = extensionsConfigPath match {
      case null | "" =>
        val path = ModelType.resolveActualExtensionsConfigPath(modelType)
        s"$openvinoTempDirPath/$path"
      case _ => extensionsConfigPath
    }

    timing("optimize tf object detection model to openvino IR") {
      val outputPath: String = outputDir
      val stdout = new StringBuilder
      val stderr = new StringBuilder
      val log = ProcessLogger(stdout append _ + "\n", stderr append _ + "\n")
      val exitCode = timing("optimize tf object detection model execution") {
        Seq("sh",
          optimizeObjectDetectionSHPath,
          modelPath,
          actualPipelineConfigPath,
          actualExtensionsConfigPath,
          outputPath,
          motfpyFilePath) ! log
      }
      logger.info(s"tf object detection model optimized, log: \n" +
        s"stderr: $stderr \n" +
        s"stdout: $stdout \n" +
        s"exitCode: $exitCode\n -----")
      exitCode match {
        case 0 => logger.info(s"tf object detection model optimization succeeded")
        case _ =>
          val message = stderr.toString().split("\n").filter(_ contains ("ERROR")).mkString(",")
          throw
            new InferenceRuntimeException(s"Openvino optimize tf object detection model error: " +
              s"$exitCode, $message")
      }
    }
  }

  def calibrateTensorflowModel(modelPath: String,
                               networkType: String,
                               validationFilePath: String,
                               subset: Int,
                               opencvLibPath: String,
                               outputDir: String): Unit = {
    logger.info(s"start to calibrate tf model from " +
      s"$modelPath, $networkType, $validationFilePath, $subset, $opencvLibPath, to $outputDir")

    timing("calibrate tf model") {
      val modelName = modelPath.split("\\/").last.split("\\.").head
      val outputPath: String = s"$outputDir/$modelName-calibrated"
      val stdout = new StringBuilder
      val stderr = new StringBuilder
      val log = ProcessLogger(stdout append _ + "\n", stderr append _ + "\n")
      val exitCode = timing("calibrate tf model execution") {
        Seq("sh",
          calibrateSHPath,
          networkType,
          modelPath,
          validationFilePath,
          subset + "",
          outputPath,
          calibrationToolPath,
          opencvLibPath,
          calibrationLibPath
        ) ! log
      }
      logger.info(s"tf model calibrated, log: \n" +
        s"stderr: $stderr \n" +
        s"stdout: $stdout \n" +
        s"exitCode: $exitCode\n -----")
      exitCode match {
        case 0 => logger.info(s"tf model calibrate succeeded")
        case _ =>
          val message = stderr.toString().split("\n").filter(_ contains ("ERROR")).mkString(",")
          throw
            new InferenceRuntimeException(s"Openvino calibrate tf model error: " +
              s"$exitCode, $message")
      }
    }
  }

  def loadTensorflowModel(modelPath: String,
                          modelType: String,
                          pipelineConfigPath: String,
                          extensionsConfigPath: String): OpenVINOModel = {
    val tmpDir = Utils.createTmpDir("ZooVino").toFile()
    val outputPath: String = tmpDir.getCanonicalPath

    optimizeTFObjectDetectionModel(modelPath, modelType,
      pipelineConfigPath, extensionsConfigPath, outputPath)

    val modelName = modelPath.split("\\/").last.split("\\.").head
    loadOpenVinoIRFromTempDir(modelName, outputPath)
  }

  def loadTensorflowModel(modelPath: String,
                          modelType: String,
                          checkpointPath: String,
                          inputShape: Array[Int],
                          ifReverseInputChannels: Boolean,
                          meanValues: Array[Float],
                          scale: Float): OpenVINOModel = {
    val tmpDir = Utils.createTmpDir("ZooVino").toFile()
    val outputPath: String = tmpDir.getCanonicalPath

    optimizeTFImageClassificationModel(modelPath, modelType,
      checkpointPath, inputShape, ifReverseInputChannels, meanValues, scale, outputPath)

    val path = ModelType.resolveActualInferenceGraphPath(modelType)
    val modelName = path.split("\\/").last.split("\\.").head
    loadOpenVinoIRFromTempDir(modelName, outputPath)
  }

  def loadTensorflowModel(modelBytes: Array[Byte],
                          modelType: String,
                          checkpointBytes: Array[Byte],
                          inputShape: Array[Int],
                          ifReverseInputChannels: Boolean,
                          meanValues: Array[Float],
                          scale: Float): OpenVINOModel = {
    val tmpDir = Utils.createTmpDir("ZooVino").toFile()
    val outputPath: String = tmpDir.getCanonicalPath
    val modelPath = (modelBytes == null) match {
      case true => null
      case false => val modelFileName = modelType + ".pb"
        val modelFile = new File(s"$tmpDir/$modelFileName")
        val modelFileInputStream = new ByteArrayInputStream(modelBytes)
        val modelFileSrc = Channels.newChannel(modelFileInputStream)
        val modelFileDest = new FileOutputStream(modelFile).getChannel
        modelFileDest.transferFrom(modelFileSrc, 0, Long.MaxValue)
        modelFileDest.close()
        modelFileSrc.close()
        modelFile.getAbsolutePath
    }
    val checkpointPath = {
      val checkpointFileName = modelType + ".ckpt"
      val checkpointFile = new File(s"$tmpDir/$checkpointFileName")
      val checkpointFileInputStream = new ByteArrayInputStream(checkpointBytes)
      val checkpointFileSrc = Channels.newChannel(checkpointFileInputStream)
      val checkpointFileDest = new FileOutputStream(checkpointFile).getChannel
      checkpointFileDest.transferFrom(checkpointFileSrc, 0, Long.MaxValue)
      checkpointFileDest.close()
      checkpointFileSrc.close()
      checkpointFile.getAbsolutePath
    }
    val model = loadTensorflowModel(modelPath, modelType,
      checkpointPath, inputShape, ifReverseInputChannels, meanValues, scale)
    s"rm -rf $tmpDir" !;
    model
  }

  def loadTensorflowModel(savedModelDir: String,
                          inputShape: Array[Int],
                          ifReverseInputChannels: Boolean,
                          meanValues: Array[Float],
                          scale: Float,
                          input: String): OpenVINOModel = {
    val tmpDir = Utils.createTmpDir("ZooVino").toFile
    val outputPath: String = tmpDir.getCanonicalPath

    optimizeTFImageClassificationModel(savedModelDir, inputShape, ifReverseInputChannels,
      meanValues, scale, input, outputPath)
    val path = new File(outputPath).listFiles()
      .filter(_.getAbsolutePath.endsWith(".mapping"))(0).getAbsolutePath
    val modelName = path.split("\\/").last.split("\\.").head
    loadOpenVinoIRFromTempDir(modelName, outputPath)
  }

  def loadTensorflowModel(savedModelBytes: Array[Byte],
                          inputShape: Array[Int],
                          ifReverseInputChannels: Boolean,
                          meanValues: Array[Float],
                          scale: Float,
                          input: String): OpenVINOModel = {
    val tmpDir = Utils.createTmpDir("ZooVino").toFile()
    val outputPath: String = tmpDir.getCanonicalPath

    val tarFilePath = (savedModelBytes == null) match {
      case true => null
      case false => val tarFileName = "saved-model.tar"
        val tarFile = new File(s"$tmpDir/$tarFileName")
        val tarFileInputStream = new ByteArrayInputStream(savedModelBytes)
        val tarFileSrc = Channels.newChannel(tarFileInputStream)
        val tarFileDest = new FileOutputStream(tarFile).getChannel
        tarFileDest.transferFrom(tarFileSrc, 0, Long.MaxValue)
        tarFileDest.close()
        tarFileSrc.close()
        tarFile.getAbsolutePath
    }
    s"mkdir -p $tmpDir/saved-model" !;
    s"tar xvf $tarFilePath -C $tmpDir/saved-model" !;
    s"ls $tmpDir/saved-model" !;
    val savedModelDir = new File(s"$tmpDir/saved-model").listFiles()(0).getAbsolutePath

    optimizeTFImageClassificationModel(savedModelDir, inputShape, ifReverseInputChannels,
      meanValues, scale, input, outputPath)
    val path = new File(outputPath).listFiles()
      .filter(_.getAbsolutePath.endsWith(".mapping"))(0).getAbsolutePath
    val modelName = path.split("\\/").last.split("\\.").head
    loadOpenVinoIRFromTempDir(modelName, outputPath)
  }

  def loadTensorflowModelAsCalibrated(modelPath: String,
                                      modelType: String,
                                      checkpointPath: String,
                                      inputShape: Array[Int],
                                      ifReverseInputChannels: Boolean,
                                      meanValues: Array[Float],
                                      scale: Float,
                                      networkType: String,
                                      validationFilePath: String,
                                      subset: Int,
                                      opencvLibPath: String): OpenVINOModel = {
    val tmpDir = Utils.createTmpDir("ZooVino").toFile()
    val outputPath: String = tmpDir.getCanonicalPath

    optimizeTFImageClassificationModel(modelPath, modelType,
      checkpointPath, inputShape, ifReverseInputChannels, meanValues, scale, outputPath)

    val path = ModelType.resolveActualInferenceGraphPath(modelType)
    val modelName = path.split("\\/").last.split("\\.").head
    calibrateOpenVinoIRFromTempDir(modelName, outputPath,
      networkType, validationFilePath, subset, opencvLibPath)

    val calibratedModelName = s"$modelName-calibrated"
    loadOpenVinoIRFromTempDir(calibratedModelName, outputPath)
  }

  def loadOpenVinoIRFromTempDir(modelName: String, tempDir: String): OpenVINOModel = {
    val modelFilePath: String = s"$tempDir/$modelName.xml"
    val weightFilePath: String = s"$tempDir/$modelName.bin"
    val mappingFilePath: String = s"$tempDir/$modelName.mapping"

    timing(s"load openvino IR from $modelFilePath, $weightFilePath, $mappingFilePath") {
      val modelFile = new File(modelFilePath)
      val weightFile = new File(weightFilePath)
      val mappingFile = new File(mappingFilePath)
      val model = (modelFile.exists(), weightFile.exists(), mappingFile.exists()) match {
        case (true, true, _) =>
          loadOpenVinoIR(modelFilePath, weightFilePath, DeviceType.CPU)
        case (_, _, _) => throw
          new InferenceRuntimeException("Openvino optimize tf model error")
      }
      timing("delete temporary model files") {
        modelFile.delete()
        weightFile.delete()
        if(mappingFile.exists()) {mappingFile.delete()}
      }
      model
    }
  }

  def calibrateOpenVinoIRFromTempDir(modelName: String,
                                     tempDir: String,
                                     networkType: String,
                                     validationFilePath: String,
                                     subset: Int,
                                     opencvLibPath: String): Unit = {
    val modelFilePath: String = s"$tempDir/$modelName.xml"
    val weightFilePath: String = s"$tempDir/$modelName.bin"
    val mappingFilePath: String = s"$tempDir/$modelName.mapping"

    timing(s"load openvino IR from $modelFilePath, $weightFilePath, $mappingFilePath") {
      val modelFile = new File(modelFilePath)
      val weightFile = new File(weightFilePath)
      val mappingFile = new File(mappingFilePath)
      val model = (modelFile.exists(), weightFile.exists(), mappingFile.exists()) match {
        case (true, true, true) =>
          calibrateTensorflowModel(modelFilePath, networkType,
            validationFilePath, subset, opencvLibPath, tempDir)
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
                     deviceType: DeviceTypeEnumVal,
                     batchSize: Int = 0): OpenVINOModel = {
    timing("load openvino IR") {
      val buffer = Source.fromFile(modelFilePath)
      val isInt8 = buffer.getLines().count(_ matches ".*statistics.*")
      buffer.close()

      val supportive: OpenVinoInferenceSupportive = new OpenVinoInferenceSupportive()
      val executableNetworkReference: Long = if (isInt8 > 0) {
        logger.info(s"Load int8 model")
        supportive.loadOpenVinoIRInt8(modelFilePath, weightFilePath,
          deviceType.value, batchSize)
      } else {
        supportive.loadOpenVinoIR(modelFilePath, weightFilePath,
          deviceType.value, batchSize)
      }
      new OpenVINOModel(executableNetworkReference, supportive, isInt8 > 0)
    }
  }

  def loadOpenVinoIR(modelBytes: Array[Byte],
                     weightBytes: Array[Byte],
                     deviceType: DeviceTypeEnumVal,
                     batchSize: Int): OpenVINOModel = {
    timing("load openvino IR") {
      val tmpDir = Utils.createTmpDir("ZooVino").toFile()
      val modelFilePath = (modelBytes == null) match {
        case true => null
        case false => val modelFileName = "model.xml"
          val modelFile = new File(s"$tmpDir/$modelFileName")
          val modelFileInputStream = new ByteArrayInputStream(modelBytes)
          val modelFileSrc = Channels.newChannel(modelFileInputStream)
          val modelFileDest = new FileOutputStream(modelFile).getChannel
          modelFileDest.transferFrom(modelFileSrc, 0, Long.MaxValue)
          modelFileDest.close()
          modelFileSrc.close()
          modelFile.getAbsolutePath
      }
      val weightFilePath = {
        val weightFileName = "weights.bin"
        val weightFile = new File(s"$tmpDir/$weightFileName")
        val weightFileInputStream = new ByteArrayInputStream(weightBytes)
        val weightFileSrc = Channels.newChannel(weightFileInputStream)
        val weightFileDest = new FileOutputStream(weightFile).getChannel
        weightFileDest.transferFrom(weightFileSrc, 0, Long.MaxValue)
        weightFileDest.close()
        weightFileSrc.close()
        weightFile.getAbsolutePath
      }

      val buffer = Source.fromFile(modelFilePath)
      val isInt8 = buffer.getLines().count(_ matches ".*statistics.*")
      buffer.close()

      val supportive: OpenVinoInferenceSupportive = new OpenVinoInferenceSupportive()
      val executableNetworkReference: Long = if (isInt8 > 0) {
        logger.info(s"Load int8 model")
        supportive.loadOpenVinoIRInt8(modelFilePath, weightFilePath,
          deviceType.value, batchSize)
      } else {
        supportive.loadOpenVinoIR(modelFilePath, weightFilePath,
          deviceType.value, batchSize)
      }
      val model = new OpenVINOModel(executableNetworkReference, supportive, isInt8 > 0)
      s"rm -rf $tmpDir" !;
      model
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

  val image_classification_types = List(
    "inception_v1",
    "inception_v2",
    "inception_v3",
    "inception_v4",
    "inception_resnet_v2",
    "mobilenet_v1",
    "nasnet_large",
    "nasnet_mobile",
    "resnet_v1_50",
    "resnet_v2_50",
    "resnet_v1_101",
    "resnet_v2_101",
    "resnet_v1_152",
    "resnet_v2_152",
    "vgg_16",
    "vgg_19"
  )

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

  def isSupportedImageClassificationModel(modelType : String): Boolean = {
    image_classification_types.contains(modelType)
  }

  def isSupportedObjectDetectionModel(modelType : String): Boolean = {
    object_detection_types.contains(modelType)
  }

  def resolveActualInferenceGraphPath(modelType : String): String = {
    s"inference-graphs/${modelType}_inference_graph.pb"
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
