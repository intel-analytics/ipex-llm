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

package com.intel.analytics.bigdl.serving.models

import com.intel.analytics.bigdl.serving.{ClusterServing, ClusterServingHelper, ClusterServingInference}
import com.intel.analytics.bigdl.serving.serialization.ArrowDeserializer
import org.scalatest.{FlatSpec, Matchers}

import scala.sys.process._

class OpenVINOModelSpec extends FlatSpec with Matchers {
  ClusterServing.helper = new ClusterServingHelper()
  "OpenVINO Inception_v1" should "work" in {
    ("wget -O /tmp/openvino_inception_v1.tar https://sourceforge.net/projects/" +
"analytics-zoo/files/analytics-zoo-data/openvino_inception_v1.tar").!
    "tar -xvf /tmp/openvino_inception_v1.tar -C /tmp/".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/image-3_224_224-arrow-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString


    val helper = ClusterServing.helper
    helper.modelType = "openvino"
    helper.weightPath = "/tmp/openvino_inception_v1/inception_v1.bin"
    helper.defPath = "/tmp/openvino_inception_v1/inception_v1.xml"

    ClusterServing.model = helper.loadInferenceModel()
    Seq("sh", "-c", "rm -rf /tmp/openvino_inception_v1*").!

    val inference = new ClusterServingInference()
    val in = List(("1", b64string, ""), ("2", b64string, ""), ("3", b64string, ""))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 1001, "result length wrong")
      require(result(0)._2.length == 1, "result shape wrong")
    })
  }

  "OpenVINO Mobilenet_v1" should "work" in {
    ("wget -O /tmp/openvino_mobilenet_v1.tar https://sourceforge.net/projects/" +
"analytics-zoo/files/analytics-zoo-data/openvino_mobilenet_v1.tar").!
    "tar -xvf /tmp/openvino_mobilenet_v1.tar -C /tmp/".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/image-3_224_224-arrow-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString

    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.modelType = "openvino"
    helper.weightPath = "/tmp/openvino_mobilenet_v1/mobilenet_v1_1.0_224_frozen.bin"
    helper.defPath = "/tmp/openvino_mobilenet_v1/mobilenet_v1_1.0_224_frozen.xml"

    ClusterServing.model = helper.loadInferenceModel()

    Seq("sh", "-c", "rm -rf /tmp/openvino_mobilenet_v1*").!

    val inference = new ClusterServingInference()
    val in = List(("1", b64string, ""), ("2", b64string, ""), ("3", b64string, ""))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 1001, "result length wrong")
      require(result(0)._2.length == 1, "result shape wrong")
    })
  }


  "OpenVINO Mobilenet_v2" should "work" in {
    ("wget -O /tmp/openvino_mobilenet_v2.tar https://sourceforge.net/projects/" +
"analytics-zoo/files/analytics-zoo-data/openvino_mobilenet_v2.tar").!
    "tar -xvf /tmp/openvino_mobilenet_v2.tar -C /tmp/".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/image-3_224_224-arrow-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString

    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.modelType = "openvino"
    helper.weightPath = "/tmp/openvino_mobilenet_v2/mobilenet_v2.bin"
    helper.defPath = "/tmp/openvino_mobilenet_v2/mobilenet_v2.xml"

    ClusterServing.model = helper.loadInferenceModel()

    Seq("sh", "-c", "rm -rf /tmp/openvino_mobilenet_v2*").!

    val inference = new ClusterServingInference()
    val in = List(("1", b64string, ""), ("2", b64string, ""), ("3", b64string, ""))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 1001, "result length wrong")
      require(result(0)._2.length == 1, "result shape wrong")
    })
  }


  "OpenVINO Resnet50_openvino2020" should "work" in {
    ("wget -O /tmp/openvino2020_resnet50.tar https://sourceforge.net/projects/" +
"analytics-zoo/files/analytics-zoo-data/openvino2020_resnet50.tar").!
    "tar -xvf /tmp/openvino2020_resnet50.tar -C /tmp/".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/image-3_224_224-arrow-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString

    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.modelType = "openvino"
    helper.weightPath = "/tmp/openvino2020_resnet50/resnet_v1_50.bin"
    helper.defPath = "/tmp/openvino2020_resnet50/resnet_v1_50.xml"

    ClusterServing.model = helper.loadInferenceModel()

    Seq("sh", "-c", "rm -rf /tmp/openvino2020_resnet50*").!

    val inference = new ClusterServingInference()
    val in = List(("1", b64string, ""), ("2", b64string, ""), ("3", b64string, ""))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 1000, "result length wrong")
      require(result(0)._2.length == 1, "result shape wrong")
    })
  }


  "OpenVINO Resnet50" should "work" in {
    ("wget -O /tmp/openvino_resnet50.tar https://sourceforge.net/projects/" +
"analytics-zoo/files/analytics-zoo-data/openvino_resnet50.tar").!
    "tar -xvf /tmp/openvino_resnet50.tar -C /tmp/".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/image-3_224_224-arrow-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString

    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.modelType = "openvino"
    helper.weightPath = "/tmp/openvino_resnet50/frozen_inference_graph.bin"
    helper.defPath = "/tmp/openvino_resnet50/frozen_inference_graph.xml"

    ClusterServing.model = helper.loadInferenceModel()

    Seq("sh", "-c", "rm -rf /tmp/openvino_resnet50*").!

    val inference = new ClusterServingInference()
    val in = List(("1", b64string, ""), ("2", b64string, ""), ("3", b64string, ""))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 1000, "result length wrong")
      require(result(0)._2.length == 1, "result shape wrong")
    })
  }



  "OpenVINO Vgg16" should "work" in {
    ("wget -O /tmp/openvino_vgg16.tar https://sourceforge.net/projects/" +
"analytics-zoo/files/analytics-zoo-data/openvino_vgg16.tar").!
    "tar -xvf /tmp/openvino_vgg16.tar -C /tmp/".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/image-3_224_224-arrow-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString

    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.modelType = "openvino"
    helper.weightPath = "/tmp/openvino_vgg16/vgg_16.bin"
    helper.defPath = "/tmp/openvino_vgg16/vgg_16.xml"

    ClusterServing.model = helper.loadInferenceModel()

    Seq("sh", "-c", "rm -rf /tmp/openvino_vgg16*").!

    val inference = new ClusterServingInference()
    val in = List(("1", b64string, ""), ("2", b64string, ""), ("3", b64string, ""))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 1000, "result length wrong")
      require(result(0)._2.length == 1, "result shape wrong")
    })
  }


  "OpenVINO face_detection_0100" should "work" in {
    ("wget -O /tmp/openvino_face_detection_0100.tar https://sourceforge.net/projects/" +
"analytics-zoo/files/analytics-zoo-data/openvino_face_detection_0100.tar").!
    "tar -xvf /tmp/openvino_face_detection_0100.tar -C /tmp/".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/image-3_224_224-arrow-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString

    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.modelType = "openvino"
    helper.weightPath = "/tmp/openvino_face_detection_0100/face-detection-0100.bin"
    helper.defPath = "/tmp/openvino_face_detection_0100/face-detection-0100.xml"

    ClusterServing.model = helper.loadInferenceModel()

    Seq("sh", "-c", "rm -rf /tmp/openvino_face_detection_0100*").!

    val inference = new ClusterServingInference()
    val in = List(("1", b64string, ""), ("2", b64string, ""), ("3", b64string, ""))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 1400, "result length wrong")
      require(result(0)._2.length == 3, "result shape wrong")
    })
  }

}
