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

import java.io.{File, FileInputStream}
import java.util
import java.util.{Arrays, Properties}

import org.scalatest._
import org.slf4j.LoggerFactory
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.common.Utils

import scala.io.Source
import scala.language.postfixOps
import scala.sys.process._
import sys.env


@OpenVinoTest
class OpenVINOInt8Suite extends FunSuite with Matchers with BeforeAndAfterAll
  with InferenceSupportive {

  val s3Url = if (env.contains("FTP_URI")) {
    env("FTP_URI").toString
  } else {
    "https://s3-ap-southeast-1.amazonaws.com"
  }

  val logger = LoggerFactory.getLogger(getClass)
  var tmpDir: File = _

  val resnet_v1_50_url = s"$s3Url/analytics-zoo-models/openvino/2018_R5/resnet_v1_50"

  val resnet_v1_50_inputShape = Array(4, 224, 224, 3)
  var resnet_v1_50_path: String = _
  var resnet_v1_50_int8_path: String = _

  val valTarUrl = s"$s3Url/analytics-zoo-models/openvino/val_bmp_32.tar"
  val valTar = valTarUrl.split("/").last
  var valDir: String = _

  val resnet_v1_50_shape = Array(4, 3, 224, 224)
  val image_input_65_url = s"$s3Url/analytics-zoo-models/openvino/ic_input_65"
  val image_input_970_url = s"$s3Url/analytics-zoo-models/openvino/ic_input_970"
  var image_input_65_filePath: String = _
  var image_input_970_filePath: String = _

  override def beforeAll() {
    System.clearProperty("bigdl.localMode")
    System.clearProperty("bigdl.coreNumber")

    tmpDir = Utils.createTmpDir("ZooVino").toFile()
    val dir = new File(s"${tmpDir.getAbsolutePath}/OpenVinoInt8Spec").getCanonicalPath

    s"wget -nv -P $dir ${resnet_v1_50_url}.xml" !;
    s"wget -nv -P $dir ${resnet_v1_50_url}.bin" !;
    s"wget -nv -P $dir ${resnet_v1_50_url}_i8.xml" !;
    s"wget -nv -P $dir ${resnet_v1_50_url}_i8.bin" !;

    s"wget -nv -P $dir $valTarUrl" !;
    s"tar xvf $dir/$valTar -C $dir" !;

    s"wget -nv -P $dir $image_input_65_url" !;
    s"wget -nv -P $dir $image_input_970_url" !;

    s"ls -alh $dir" !;

    valDir = s"$dir/val_bmp_32/"

    image_input_65_filePath = s"$dir/ic_input_65"
    image_input_970_filePath = s"$dir/ic_input_970"

    // int8 optimized model
    resnet_v1_50_path = s"${dir}/resnet_v1_50"

    resnet_v1_50_int8_path = s"${dir}/resnet_v1_50_i8"
    tmpDir.listFiles().foreach(file => println(file.getAbsoluteFile))

  }

  override def afterAll() {
    // FileUtils.deleteDirectory(tmpDir)
    s"rm -rf $tmpDir" !;
  }


  test("openvino should load from bytes of IR") {
    val model = new AbstractInferenceModel() {
    }

    val modelFilePath = s"${resnet_v1_50_int8_path}.xml"
    val weightFilePath = s"${resnet_v1_50_int8_path}.bin"
    val batchSize = resnet_v1_50_inputShape.apply(0)
    val modelFileSize = new File(modelFilePath).length()
    val modelFileInputStream = new FileInputStream(modelFilePath)
    val modelFileBytes = new Array[Byte](modelFileSize.toInt)
    modelFileInputStream.read(modelFileBytes)

    val weightFileSize = new File(weightFilePath).length()
    val weightFileInputStream = new FileInputStream(weightFilePath)
    val weightFileBytes = new Array[Byte](weightFileSize.toInt)
    weightFileInputStream.read(weightFileBytes)

    model.loadOpenVINO(modelFileBytes, weightFileBytes, batchSize)

    println(model)
  }


  test("openvino doLoadOpenVINO(float) and predict(float)") {
    val model = new InferenceModel(3)
    model.doLoadOpenVINO(s"${resnet_v1_50_path}.xml",
      s"${resnet_v1_50_path}.bin")
    println(s"resnet_v1_50_model from tf loaded as $model")
    model shouldNot be(null)

    val indata1 = fromHWC2CHW(Source.fromFile(image_input_65_filePath)
      .getLines().map(_.toFloat).toArray)
    val indata2 = fromHWC2CHW(Source.fromFile(image_input_970_filePath)
      .getLines().map(_.toFloat).toArray)
    val labels = Array(65f, 795f)
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
      println(s"(${class1}, ${class2})")
      Array(class1.toFloat, class2.toFloat)
    })
    classes.foreach { output =>
      assert(almostEqual(output, labels, 0.1f))
    }
  }

  test("openvino doLoadInt8 and PredictInt8(float)") {
    val model = new InferenceModel(3)
    model.doLoadOpenVINO(s"${resnet_v1_50_int8_path}.xml",
      s"${resnet_v1_50_int8_path}.bin",
      resnet_v1_50_inputShape.apply(0))
    println(s"resnet_v1_50_model from tf loaded as $model")
    model shouldNot be(null)
    val indata1 = fromHWC2CHW(Source.fromFile(image_input_65_filePath)
      .getLines().map(_.toFloat).toArray)
    val indata2 = fromHWC2CHW(Source.fromFile(image_input_970_filePath)
      .getLines().map(_.toFloat).toArray)
    // 65's top1 is 65, 970's top1 is 795
    val labels = Array(65f, 795f)
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

    // Load float32 and predict int8 leads to 65 to wrong results
    // PredictInt8
    val resultsInt8: util.List[util.List[JTensor]] = model.doPredict(inputs)
    val classesInt8 = resultsInt8.toArray().map(list => {
      val inner = list.asInstanceOf[util.List[JTensor]].get(0)
      val class1 = inner.getData.slice(0, 1000).zipWithIndex.maxBy(_._1)._2
      val class2 = inner.getData.slice(1000, 2000).zipWithIndex.maxBy(_._1)._2
      println(s"(${class1}, ${class2})")
      Array(class1.toFloat, class2.toFloat)
    })
    classesInt8.foreach { output =>
      assert(almostEqual(output, labels, 0.1f))
    }
  }

  test("openvino resnet50 should predict image successfully") {
    val model = new InferenceModel(3)
    model.doLoadOpenVINO(s"${resnet_v1_50_int8_path}.xml",
      s"${resnet_v1_50_int8_path}.bin")
    println(s"resnet_v1_50_model from tf loaded as $model")
    model shouldNot be(null)

    var indata1 = new Array[Float](3 * 224 * 224)
    var indata2 = new Array[Float](3 * 224 * 224)
    OpenCVMat.toFloatPixels(OpenCVMat.read(valDir +
      "ILSVRC2012_val_00000001.bmp"), indata1)
    OpenCVMat.toFloatPixels(OpenCVMat.read(valDir +
      "ILSVRC2012_val_00000002.bmp"), indata2)
    indata1 = fromHWC2CHW(indata1)
    indata2 = fromHWC2CHW(indata2)
    val labels = Array(65f, 795f)
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

    // PredictInt8
    val results: util.List[util.List[JTensor]] = model.doPredict(inputs)
    val classes = results.toArray().map(list => {
      val inner = list.asInstanceOf[util.List[JTensor]].get(0)
      val class1 = inner.getData.slice(0, 1000).zipWithIndex.maxBy(_._1)._2
      val class2 = inner.getData.slice(1000, 2000).zipWithIndex.maxBy(_._1)._2
      println(s"(${class1}, ${class2})")
      Array(class1.toFloat, class2.toFloat)
    })
    classes.foreach { output =>
      assert(almostEqual(output, labels, 0.1f))
    }
  }

  test("openvino should handle wrong batchSize correctly") {
    val model = new InferenceModel(3)
    model.doLoadOpenVINO(s"${resnet_v1_50_int8_path}.xml",
      s"${resnet_v1_50_int8_path}.bin",
      resnet_v1_50_inputShape.apply(0))
    println(s"resnet_v1_50_model from tf loaded as $model")
    model shouldNot be(null)
    val indata1 = fromHWC2CHW(Source.fromFile(image_input_65_filePath)
      .getLines().map(_.toFloat).toArray)
    val indata2 = fromHWC2CHW(Source.fromFile(image_input_970_filePath)
      .getLines().map(_.toFloat).toArray)
    val labels = Array(65f, 795f)
    // batchSize = 4, but given 3 and 5
    val data1 = indata1 ++ indata2 ++ indata1
    val data2 = indata2 ++ indata1 ++ indata2 ++ indata1 ++ indata2
    val input1 = new JTensor(data1, Array(3, 3, 224, 224))
    val input2 = new JTensor(data2, Array(5, 3, 224, 224))
    val inputs = Arrays.asList(
      Arrays.asList({
        input1
      }),
      Arrays.asList({
        input2
      }))

    val resultsInt8: util.List[util.List[JTensor]] = model.doPredict(inputs)
    val classesInt8 = resultsInt8.toArray().map(list => {
      val inner = list.asInstanceOf[util.List[JTensor]].get(0)
      val class1 = inner.getData.slice(0, 1000).zipWithIndex.maxBy(_._1)._2
      class1.toFloat
    })
    assert(almostEqual(classesInt8, labels, 0.1f))
    println(classesInt8.mkString(","))
  }

  def fromHWC2CHW(data: Array[Float]): Array[Float] = {
    val resArray = new Array[Float](3 * 224 * 224)
    for (h <- 0 to 223) {
      for (w <- 0 to 223) {
        for (c <- 0 to 2) {
          resArray(c * 224 * 224 + h * 224 + w) = data(h * 224 * 3 + w * 3 + c)
        }
      }
    }
    resArray
  }

  def fromCHW2HWC(data: Array[Float]): Array[Float] = {
    val resArray = new Array[Float](3 * 224 * 224)
    for (c <- 0 to 2) {
      for (h <- 0 to 223) {
        for (w <- 0 to 223) {
          resArray(h * 224 * 3 + w * 3 + c) = data(c * 224 * 224 + h * 224 + w)
        }
      }
    }
    resArray
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
