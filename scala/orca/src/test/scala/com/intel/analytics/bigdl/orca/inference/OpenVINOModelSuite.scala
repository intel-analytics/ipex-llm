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

import java.io.File
import java.nio.file.NoSuchFileException
import java.util
import java.util.{Arrays, Properties}

import com.intel.analytics.zoo.common.Utils
import org.codehaus.plexus.util.FileUtils
import org.scalatest._
import org.slf4j.LoggerFactory

import scala.io.Source
import scala.language.postfixOps
import sys.process._
import sys.env


@OpenVinoTest
class OpenVINOModelSuite extends FunSuite with Matchers with BeforeAndAfterAll
  with InferenceSupportive {

  val s3Url = if (env.contains("FTP_URI")) {
    env("FTP_URI").toString
  } else {
    "https://s3-ap-southeast-1.amazonaws.com"
  }
  val s3DataUrl = s"$s3Url" +
    s"/analytics-zoo-models/openvino/Tests_faster_rcnn_resnet101_coco_2018_01_28"
  val url_ov_fasterrcnn_tests_inputdata1 = s"$s3DataUrl/inputdata_1"
  val url_ov_fasterrcnn_tests_inputdata2 = s"$s3DataUrl/inputdata_2"

  val logger = LoggerFactory.getLogger(getClass)
  var tmpDir: File = _

  val fasterrcnnModelUrl = s"$s3Url" +
    s"/analytics-zoo-models/openvino/2018_R5/faster_rcnn_resnet101_coco"
  var fasterrcnnModel: OpenVINOModel = _
  val fasterrcnnInferenceModel: InferenceModel = new InferenceModel(3)
  val fasterrcnnInputShape = Array(1, 3, 600, 600)
  var faserrcnnModelPath: String = _
  var fasterrcnnInputdata1FilePath: String = _
  var fasterrcnnInputdata2FilePath: String = _


  override def beforeAll() {
    System.clearProperty("bigdl.localMode")
    System.clearProperty("bigdl.coreNumber")

    tmpDir = Utils.createTmpDir("ZooVino").toFile()
    val dir = new File(s"${tmpDir.getAbsolutePath}/OpenVinoInferenceModelSpec").getCanonicalPath

    s"wget -nv -P $dir $url_ov_fasterrcnn_tests_inputdata1" !;
    s"wget -nv -P $dir $url_ov_fasterrcnn_tests_inputdata2" !;

    s"wget -nv -P $dir $fasterrcnnModelUrl.xml" !;
    s"wget -nv -P $dir $fasterrcnnModelUrl.bin" !;

    s"ls -alh $dir" !;

    faserrcnnModelPath = s"$dir/faster_rcnn_resnet101_coco"
    fasterrcnnInputdata1FilePath = s"$dir/inputdata_1"
    fasterrcnnInputdata2FilePath = s"$dir/inputdata_2"
  }

  override def afterAll() {
    // FileUtils.deleteDirectory(tmpDir)
    s"rm -rf $tmpDir" !;
  }

  test("openvino model should be optimized") {
    fasterrcnnInferenceModel
      .doLoadOpenVINO(s"$faserrcnnModelPath.xml",
        s"$faserrcnnModelPath.bin")
    tmpDir.listFiles().foreach(file => println(file.getAbsoluteFile))
  }

  test("openvino model should throw exception if load failed") {
    val thrown = intercept[NoSuchFileException] {
      InferenceModelFactory
        .loadOpenVINOModelForIR(s"$faserrcnnModelPath.error.xml",
        s"$faserrcnnModelPath.error.bin",
        DeviceType.CPU)
    }
  }

  // this method will be deprecated", "0.8.0")
  test("openvino object detection model should load successfully and predict correctly") {
    fasterrcnnModel = InferenceModelFactory
      .loadOpenVINOModelForIR(s"$faserrcnnModelPath.xml",
        s"$faserrcnnModelPath.bin",
        DeviceType.CPU)
    fasterrcnnInferenceModel
      .doLoadOpenVINO(s"$faserrcnnModelPath.xml",
        s"$faserrcnnModelPath.bin")

    println(s"fasterrcnnModel from tensorflow pb loaded as $fasterrcnnModel")
    fasterrcnnModel shouldNot be(null)
    println(s"fasterrcnnInferenceModel from tensorflow pb loaded as $fasterrcnnInferenceModel")
    fasterrcnnInferenceModel shouldNot be(null)

    val indata1 = Source.fromFile(fasterrcnnInputdata1FilePath).getLines().map(_.toFloat).toArray
    val indata2 = Source.fromFile(fasterrcnnInputdata2FilePath).getLines().map(_.toFloat).toArray
    println(indata1.length, indata2.length, 1 * 3 * 600 * 600)
    val input1 = new JTensor(indata1, fasterrcnnInputShape)
    val input2 = new JTensor(indata2, fasterrcnnInputShape)
    val inputs = Arrays.asList(
      Arrays.asList({
        input1
      }),
      Arrays.asList({
        input2
      }))

    val results2 = fasterrcnnModel.predict(inputs)
    val results4 = fasterrcnnInferenceModel.doPredict(inputs)

    val threads2 = List.range(0, 5).map(i => {
      new Thread() {
        override def run(): Unit = {
          val results = fasterrcnnInferenceModel.doPredict(inputs)
        }
      }
    })
    threads2.foreach(_.start())
    threads2.foreach(_.join())

    fasterrcnnModel.release()
    fasterrcnnInferenceModel.doRelease()
  }

  def almostEqual(x: Float, y: Float, precision: Float): Boolean = {
    (x - y).abs <= precision match {
      case true => true
      case false => println(x, y); false
    }
  }

  def almostEqual(x: Array[Float], y: Array[Float], precision: Float): Boolean = {
    x.length == y.length match {
      case true => x.zip(y).filter(t => !almostEqual(t._1, t._2, precision)).length == 0
      case false => println(x.length, y.length); false
    }
  }
}
