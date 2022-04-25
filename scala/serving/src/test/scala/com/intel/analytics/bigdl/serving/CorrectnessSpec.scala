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

package com.intel.analytics.bigdl.serving

import java.awt._
import java.awt.image.BufferedImage
import java.io.{ByteArrayOutputStream, File}
import java.util.Base64

import com.intel.analytics.bigdl.dllib.feature.image.OpenCVMethod
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.serving.http.{Instances, JsonUtil}
import com.intel.analytics.bigdl.serving.postprocessing.PostProcessing
import com.intel.analytics.bigdl.serving.utils.DeprecatedUtils
import javax.imageio.ImageIO
import org.apache.commons.io.FileUtils
import org.apache.logging.log4j.LogManager
import org.opencv.core._
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.scalatest.{FlatSpec, Matchers}

import scala.io.Source
import scala.sys.process._

class CorrectnessSpec extends FlatSpec with Matchers {
  val configPath = "/tmp/config.yaml"
  var redisHost: String = "localhost"
  var redisPort: Int = 6379
  val logger = LogManager.getLogger(getClass)
  def resize(p: String): String = {
    val source = ImageIO.read(new File(p))
    val outputImage: BufferedImage = new BufferedImage(224, 224, source.getType)
    val graphics2D: Graphics2D = outputImage.createGraphics

    graphics2D.drawImage(source, 0, 0, 224, 224, null)
    graphics2D.dispose()

    val byteStream = new ByteArrayOutputStream()
    ImageIO.write(outputImage, "jpg", byteStream)

//    val f = new File("/home/litchy/tmp/034.jpg")
//    ImageIO.write(outputImage, "jpg", f)
    val dataStr = Base64.getEncoder.encodeToString(byteStream.toByteArray)
    dataStr
  }
  def getBase64FromPath(path: String): String = {

    val b = FileUtils.readFileToByteArray(new File(path))
    val img = OpenCVMethod.fromImageBytes(b, Imgcodecs.CV_LOAD_IMAGE_COLOR)
    Imgproc.resize(img, img, new Size(224, 224))
    val matOfByte = new MatOfByte()
    Imgcodecs.imencode(".jpg", img, matOfByte)
    val dataStr = Base64.getEncoder.encodeToString(matOfByte.toArray)
    dataStr
  }


//  def runServingBg(): Future[Unit] = Future {
//    ClusterServing.run(configPath, redisHost, redisPort)
//  }
  "Cluster Serving result" should "be correct" in {

    ("wget --no-check-certificate -O /tmp/serving_val.tar https://sourceforge.net/" +
"projects/analytics-zoo/files/analytics-zoo-data/serving_val.tar").!
    "tar -xvf /tmp/serving_val.tar -C /tmp/".!
    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.configPath = configPath
    DeprecatedUtils.loadConfig(helper)
//    helper.dataShape = Array(Array(3, 224, 224))
    val model = helper.loadInferenceModel()
    val imagePath = "/tmp/imagenet_1k"
    val lsCmd = "ls " + imagePath

    val totalNum = (lsCmd #| "wc").!!.split(" +").filter(_ != "").head.toInt

    // enqueue image
    val f = new File(imagePath)
    val fileList = f.listFiles
    logger.info(s"${fileList.size} images about to enqueue...")

    val clusterServingInference = new ClusterServingInference()
    var predictMap = Map[String, String]()

    for (file <- fileList) {
      val dataStr = getBase64FromPath(file.getAbsolutePath)
      val instancesJson =
       s"""{
         |"instances": [
         |   {
         |     "img": "${dataStr}"
         |   }
         |]
         |}
         |""".stripMargin
      val instances = JsonUtil.fromJson(classOf[Instances], instancesJson)
      val inputBase64 = new String(java.util.Base64.getEncoder
       .encode(instances.toArrow()))
      val input = clusterServingInference.preProcessing.decodeArrowBase64("", inputBase64)
      val bInput = clusterServingInference.batchInput(Seq(("", input)), 1, true, false)
      val result = model.doPredict(bInput)
      val value = PostProcessing(result.toTensor[Float], "topN(1)", 1)
      val clz = value.split(",")(0).stripPrefix("[[")
      predictMap = predictMap + (file.getName -> clz)
    }
    ("rm -rf /tmp/" + imagePath + "*").!
    "rm -rf /tmp/serving_val_*".!
    "rm -rf /tmp/config.yaml".!

    // start check with txt file

    var cN = 0f
    var tN = 0f
    for (line <- Source.fromFile(imagePath + ".txt").getLines()) {
     val key = line.split(" ").head
     val cls = line.split(" ").tail(0)
     try {
       if (predictMap(key) == cls) {
         cN += 1
       }
       tN += 1
     }
     catch {
       case _ => None
     }
    }
    val acc = cN / tN
    logger.info(s"Top 1 Accuracy of serving, Openvino ResNet50 Model on ImageNet is ${acc}")
    assert(acc > 0.71)

  }

  "Cluster Serving batch inference result" should "be correct" in {

    ("wget --no-check-certificate -O /tmp/serving_val.tar https://sourceforge.net/" +
"projects/analytics-zoo/files/analytics-zoo-data/serving_val.tar").!
    "tar -xvf /tmp/serving_val.tar -C /tmp/".!
    ClusterServing.helper = new ClusterServingHelper()
val helper = ClusterServing.helper
    helper.configPath = configPath
    DeprecatedUtils.loadConfig(helper)
    //    helper.dataShape = Array(Array(3, 224, 224))
    val model = helper.loadInferenceModel()
    val imagePath = "/tmp/imagenet_1k"
    val lsCmd = "ls " + imagePath

    val totalNum = (lsCmd #| "wc").!!.split(" +").filter(_ != "").head.toInt

    // enqueue image
    val f = new File(imagePath)
    val fileList = f.listFiles
    logger.info(s"${fileList.size} images about to enqueue...")


    val clusterServingInference = new ClusterServingInference()
    var predictMap = Map[String, String]()

    var batchInputs = Seq[(String, Activity)]()

    for (file <- fileList) {
      val dataStr = getBase64FromPath(file.getAbsolutePath)
      val instancesJson =
        s"""{
           |"instances": [
           |   {
           |     "img": "${dataStr}"
           |   }
           |]
           |}
           |""".stripMargin
      val instances = JsonUtil.fromJson(classOf[Instances], instancesJson)
      val inputBase64 = new String(java.util.Base64.getEncoder
        .encode(instances.toArrow()))
      val input = clusterServingInference.preProcessing.decodeArrowBase64("", inputBase64)

      if (batchInputs.length < 4) {
        batchInputs = batchInputs :+ (file.getName(), input)
      }
      if (batchInputs.length == 4) {
        val bInput = clusterServingInference.batchInput(batchInputs, 4, true, false)
        val result = model.doPredict(bInput)
        var i = 0
        for ( i <- 1 to 4) {
          val value = PostProcessing(result.toTensor[Float], "topN(1)", i)
          val clz = value.split(",")(0).stripPrefix("[[")
          predictMap = predictMap + (batchInputs(i-1)._1 -> clz)
        }

        batchInputs = Seq[(String, Activity)]()
      }
    }
    ("rm -rf /tmp/" + imagePath + "*").!
    "rm -rf /tmp/serving_val_*".!
    "rm -rf /tmp/config.yaml".!

    // start check with txt file

    var cN = 0f
    var tN = 0f
    for (line <- Source.fromFile(imagePath + ".txt").getLines()) {
      val key = line.split(" ").head
      val cls = line.split(" ").tail(0)
      try {
        if (predictMap(key) == cls) {
          cN += 1
        }
        tN += 1
      }
      catch {
        case _ => None
      }
    }
    val acc = cN / tN
    logger.info(s"Top 1 Accuracy of serving, Openvino ResNet50 Model on ImageNet is ${acc}")
    assert(acc > 0.71)

  }
}
