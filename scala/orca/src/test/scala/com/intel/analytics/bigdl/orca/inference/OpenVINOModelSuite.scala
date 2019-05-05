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
import java.util
import java.util.{Arrays, Properties}

import com.google.common.io.Files
import org.codehaus.plexus.util.FileUtils
import org.scalatest._
import org.slf4j.LoggerFactory

import scala.io.Source
import scala.language.postfixOps
import sys.process._

@OpenVinoTest
class OpenVINOModelSuite extends FunSuite with Matchers with BeforeAndAfterAll
  with InferenceSupportive {

  val s3Url = "https://s3-ap-southeast-1.amazonaws.com"
  val s3DataUrl = s"$s3Url" +
    s"/analytics-zoo-models/openvino/Tests_faster_rcnn_resnet101_coco_2018_01_28"
  val url_ov_fasterrcnn_tests_inputdata1 = s"$s3DataUrl/inputdata_1"
  val url_ov_fasterrcnn_tests_inputdata2 = s"$s3DataUrl/inputdata_2"

  var modelZooUrl = "http://download.tensorflow.org"
  try {
    val prop = new Properties()
    prop.load(this.getClass.getResourceAsStream("/app.properties"))
    modelZooUrl = prop.getProperty("data-store-url")
  } catch {
    case e: Exception =>
      modelZooUrl = "http://download.tensorflow.org"
  }

  val logger = LoggerFactory.getLogger(getClass)
  var tmpDir: File = _

  val fasterrcnnModelUrl = s"$modelZooUrl" +
    s"/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz"
  val fasterrcnnModelTar = fasterrcnnModelUrl.split("/").last
  val fasterrcnnModelDir = fasterrcnnModelTar.replaceAll(".tar.gz", "")
  var fasterrcnnModel: OpenVINOModel = _
  val fasterrcnnInferenceModel: InferenceModel = new InferenceModel(3)
  val fasterrcnnInputShape = Array(1, 3, 600, 600)
  var faserrcnnFrozenModelFilePath: String = _
  var faserrcnnModelType: String = _
  var faserrcnnPipelineConfigFilePath: String = _
  var fasterrcnnInputdata1FilePath: String = _
  var fasterrcnnInputdata2FilePath: String = _

  val resnet_v1_50_url = s"$modelZooUrl" + s"/models/resnet_v1_50_2016_08_28.tar.gz"
  val resnet_v1_50_tar = resnet_v1_50_url.split("/").last
  val resnet_v1_50_dir = resnet_v1_50_tar.replaceAll(".tar.gz", "")
  val resnet_v1_50_modelType = "resnet_v1_50"
  var resnet_v1_50_checkpointPath: String = _
  val resnet_v1_50_inputShape = Array(4, 224, 224, 3)
  val resnet_v1_50_ifReverseInputChannels = true
  val resnet_v1_50_meanValues = Array(123.68f, 116.78f, 103.94f)
  val resnet_v1_50_scale = 1.0f

  val calibrateValTarUrl = s"$s3Url/analytics-zoo-models/openvino/val_bmp_32.tar"
  val calibrateValTar = calibrateValTarUrl.split("/").last
  var calibrateValFilePath: String = _

  val resnet_v1_50_shape = Array(4, 3, 224, 224)
  val image_input_65_url = s"$s3Url/analytics-zoo-models/openvino/ic_input_65"
  val image_input_970_url = s"$s3Url/analytics-zoo-models/openvino/ic_input_970"
  var image_input_65_filePath: String = _
  var image_input_970_filePath: String = _

  val opencvLibTarURL = s"$s3Url/analytics-zoo-models/openvino/opencv_4.0.0_ubuntu_lib.tar"
  val opencvLibTar = opencvLibTarURL.split("/").last
  var opencvLibPath: String = _

  override def beforeAll() {
    tmpDir = Files.createTempDir()
    val dir = new File(s"${tmpDir.getAbsolutePath}/OpenVinoInferenceModelSpec").getCanonicalPath

    s"wget -P $dir $url_ov_fasterrcnn_tests_inputdata1" !;
    s"wget -P $dir $url_ov_fasterrcnn_tests_inputdata2" !;

    s"wget -P $dir $resnet_v1_50_url" !;
    s"tar xvf $dir/$resnet_v1_50_tar -C $dir" !;

    s"wget -P $dir $fasterrcnnModelUrl" !;
    s"tar xvf $dir/$fasterrcnnModelTar -C $dir" !;

    s"wget -P $dir $calibrateValTarUrl" !;
    s"tar xvf $dir/$calibrateValTar -C $dir" !;

    s"wget -P $dir $image_input_65_url" !;
    s"wget -P $dir $image_input_970_url" !;

    s"wget -P $dir $opencvLibTarURL" !;
    s"tar xvf $dir/$opencvLibTar -C $dir" !;

    s"ls -alh $dir" !;

    resnet_v1_50_checkpointPath = s"$dir/resnet_v1_50.ckpt"
    calibrateValFilePath = s"$dir/val_bmp_32/val.txt"

    faserrcnnFrozenModelFilePath = s"$dir/$fasterrcnnModelDir/frozen_inference_graph.pb"
    faserrcnnModelType = "faster_rcnn_resnet101_coco"
    faserrcnnPipelineConfigFilePath = s"$dir/$fasterrcnnModelDir/pipeline.config"
    fasterrcnnInputdata1FilePath = s"$dir/inputdata_1"
    fasterrcnnInputdata2FilePath = s"$dir/inputdata_2"

    image_input_65_filePath = s"$dir/ic_input_65"
    image_input_970_filePath = s"$dir/ic_input_970"

    opencvLibPath = s"$dir/lib"
  }

  override def afterAll() {
    // FileUtils.deleteDirectory(tmpDir)
    s"rm -rf $tmpDir" !;
  }

  test("openvino model should be optimized") {
    InferenceModel.doOptimizeTF(
      faserrcnnFrozenModelFilePath,
      faserrcnnModelType,
      faserrcnnPipelineConfigFilePath,
      null,
      tmpDir.getAbsolutePath
    )
    tmpDir.listFiles().foreach(file => println(file.getAbsoluteFile))
  }

  test("openvino model should be optimized and calibrated") {
    InferenceModel.doOptimizeTF(
      null,
      resnet_v1_50_modelType,
      resnet_v1_50_checkpointPath,
      resnet_v1_50_inputShape,
      resnet_v1_50_ifReverseInputChannels,
      resnet_v1_50_meanValues,
      resnet_v1_50_scale,
      tmpDir.getAbsolutePath
    )
    val model_IR_path = s"${tmpDir.getAbsolutePath}/${resnet_v1_50_modelType}_inference_graph.xml"
    InferenceModel.doCalibrateTF(
      model_IR_path,
      "C",
      calibrateValFilePath,
      32,
      opencvLibPath,
      tmpDir.getAbsolutePath
    )
    tmpDir.listFiles().foreach(file => println(file.getAbsoluteFile))
  }

  test("openvino model should throw exception if load failed") {
    val thrown = intercept[InferenceRuntimeException] {
      InferenceModelFactory.loadOpenVINOModelForTF(
        faserrcnnFrozenModelFilePath + "error",
        faserrcnnModelType,
        faserrcnnPipelineConfigFilePath,
        null)
    }
    assert(thrown.getMessage.contains("Openvino optimize tf object detection model error"))
  }

  test("openvino object dection model should load successfully and predict correctly") {
    fasterrcnnModel = InferenceModelFactory.loadOpenVINOModelForTF(
      faserrcnnFrozenModelFilePath,
      faserrcnnModelType,
      faserrcnnPipelineConfigFilePath,
      null)
    fasterrcnnInferenceModel.doLoadTF(
      faserrcnnFrozenModelFilePath,
      faserrcnnModelType,
      faserrcnnPipelineConfigFilePath,
      null
    )

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
    fasterrcnnInferenceModel.release()
  }

  test("openvino image classification model should load successfully") {
    val model = new InferenceModel(3)
    model.doLoadTF(null,
      resnet_v1_50_modelType,
      resnet_v1_50_checkpointPath,
      resnet_v1_50_inputShape,
      resnet_v1_50_ifReverseInputChannels,
      resnet_v1_50_meanValues,
      resnet_v1_50_scale)
    println(s"resnet_v1_50_model from tf loaded as $model")
    model shouldNot be(null)

    val indata1 = Source.fromFile(image_input_65_filePath).getLines().map(_.toFloat).toArray
    val indata2 = Source.fromFile(image_input_970_filePath).getLines().map(_.toFloat).toArray
    val data = indata1 ++ indata2 ++ indata1 ++ indata2
    val input1 = new JTensor(data, resnet_v1_50_shape)
    val input2 = new JTensor(data, resnet_v1_50_shape)
    val inputs = Arrays.asList(
      Arrays.asList({
        input1
      }),
      Arrays.asList({
        input2
      }))
    val results: util.List[util.List[JTensor]] = model.doPredict(inputs)
    val classes = results.toArray().map(list => {
      val inner = list.asInstanceOf[util.List[JTensor]].get(0)
      val class1 = inner.getData.slice(0, 1000).zipWithIndex.maxBy(_._1)._2
      val class2 = inner.getData.slice(1000, 2000).zipWithIndex.maxBy(_._1)._2
      (class1, class2)
    })
    println(classes.mkString(","))
  }

  test("openvino image classification model should load as calibrated successfully") {
    val model = new InferenceModel(3)
    model.doLoadTFAsCalibratedOpenVINO(null,
      resnet_v1_50_modelType,
      resnet_v1_50_checkpointPath,
      resnet_v1_50_inputShape,
      resnet_v1_50_ifReverseInputChannels,
      resnet_v1_50_meanValues,
      resnet_v1_50_scale,
      "C",
      calibrateValFilePath,
      32,
      opencvLibPath)
    println(s"resnet_v1_50_model from tf loaded as $model")
    model shouldNot be(null)
    val indata1 = Source.fromFile(image_input_65_filePath).getLines().map(_.toFloat).toArray
    val indata2 = Source.fromFile(image_input_970_filePath).getLines().map(_.toFloat).toArray
    val data = indata1 ++ indata2 ++ indata1 ++ indata2
    val input1 = new JTensor(data, resnet_v1_50_shape)
    val input2 = new JTensor(data, resnet_v1_50_shape)
    val inputs = Arrays.asList(
      Arrays.asList({
        input1
      }),
      Arrays.asList({
        input2
      }))
    val results: util.List[util.List[JTensor]] = model.doPredict(inputs)
    val classes = results.toArray().map(list => {
      val inner = list.asInstanceOf[util.List[JTensor]].get(0)
      val class1 = inner.getData.slice(0, 1000).zipWithIndex.maxBy(_._1)._2
      val class2 = inner.getData.slice(1000, 2000).zipWithIndex.maxBy(_._1)._2
      (class1, class2)
    })
    println(classes.mkString(","))
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
