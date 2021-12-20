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

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.orca.inference.InferenceModel
import com.intel.analytics.bigdl.serving.{ClusterServing, ClusterServingHelper, ClusterServingInference}
import com.intel.analytics.bigdl.serving.serialization.ArrowDeserializer
import org.scalatest.{FlatSpec, Matchers}

import scala.sys.process._

class TensorflowModelSpec extends FlatSpec with Matchers {
  ClusterServing.helper = new ClusterServingHelper()
  "Tensorflow Inception v1" should "work" in {
    ("wget --no-check-certificate -O /tmp/tensorflow_inception_v1.tar https://sourceforge.net/" +
"projects/analytics-zoo/files/analytics-zoo-data/tensorflow_inception_v1.tar").!
    "mkdir /tmp/tensorflow_inception_v1/".!
    "tar -xvf /tmp/tensorflow_inception_v1.tar -C /tmp/tensorflow_inception_v1/".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/image-3_224_224-arrow-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString

    val helper = ClusterServing.helper
    helper.chwFlag = false
    helper.modelType = "tensorflowFrozenModel"
    helper.weightPath = "/tmp/tensorflow_inception_v1/"
    ClusterServing.model = helper.loadInferenceModel()

    Seq("sh", "-c", "rm -rf /tmp/tensorflow_inception_v1*").!

    val inference = new ClusterServingInference()
    val in = List(("1", b64string, ""), ("2", b64string, ""), ("3", b64string, ""))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 1001, "result length wrong")
      require(result(0)._2.length == 1, "result shape wrong")
    })
  }

  "Tensorflow MobileNet v1" should "work" in {
    ("wget --no-check-certificate -O /tmp/tensorflow_mobilenet_v1.tar https://sourceforge.net/" +
"projects/analytics-zoo/files/analytics-zoo-data/tensorflow_mobilenet_v1.tar").!
    "mkdir /tmp/tensorflow_mobilenet_v1/".!
    "tar -xvf /tmp/tensorflow_mobilenet_v1.tar -C /tmp/tensorflow_mobilenet_v1".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/image-3_224_224-arrow-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString

    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.chwFlag = false
    helper.modelType = "tensorflowFrozenModel"
    helper.weightPath = "/tmp/tensorflow_mobilenet_v1/"
    ClusterServing.model = helper.loadInferenceModel()

    Seq("sh", "-c", "rm -rf /tmp/tensorflow_mobilenet_v1*").!

    val inference = new ClusterServingInference()
    val in = List(("1", b64string, ""), ("2", b64string, ""), ("3", b64string, ""))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 1001, "result length wrong")
      require(result(0)._2.length == 1, "result shape wrong")
    })
  }


  "TensorflowModel MobileNet v2" should "work" in {
    ("wget --no-check-certificate -O /tmp/tensorflow_mobilenet_v2.tar https://sourceforge.net/" +
"projects/analytics-zoo/files/analytics-zoo-data/tensorflow_mobilenet_v2.tar").!
    "mkdir /tmp/tensorflow_mobilenet_v2/".!
    "tar -xvf /tmp/tensorflow_mobilenet_v2.tar -C /tmp/tensorflow_mobilenet_v2".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/image-3_224_224-arrow-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString

    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.chwFlag = false
    helper.modelType = "tensorflowFrozenModel"
    helper.weightPath = "/tmp/tensorflow_mobilenet_v2/"
    ClusterServing.model = helper.loadInferenceModel()

    Seq("sh", "-c", "rm -rf /tmp/tensorflow_mobilenet_v2*").!

    val inference = new ClusterServingInference()
    val in = List(("1", b64string, ""), ("2", b64string, ""), ("3", b64string, ""))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 1001, "result length wrong")
      require(result(0)._2.length == 1, "result shape wrong")
    })
  }


  "TensorflowModel ResNet 50" should "work" in {
    ("wget --no-check-certificate -O /tmp/tensorflow_resnet50.tar https://sourceforge.net/" +
"projects/analytics-zoo/files/analytics-zoo-data/tensorflow_resnet50.tar").!
    "mkdir /tmp/tensorflow_resnet50/".!
    "tar -xvf /tmp/tensorflow_resnet50.tar -C /tmp/tensorflow_resnet50".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/image-3_224_224-arrow-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString

    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.chwFlag = false
    helper.modelType = "tensorflowFrozenModel"
    helper.weightPath = "/tmp/tensorflow_resnet50/"
    ClusterServing.model = helper.loadInferenceModel()

    Seq("sh", "-c", "rm -rf /tmp/tensorflow_resnet50*").!

    val inference = new ClusterServingInference()
    val in = List(("1", b64string, ""), ("2", b64string, ""), ("3", b64string, ""))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 1000, "result length wrong")
      require(result(0)._2.length == 1, "result shape wrong")
    })
  }

  "TensorflowModel tf auto" should "work" in {
    ("wget --no-check-certificate -O /tmp/tensorflow_tfauto.tar https://sourceforge.net/" +
"projects/analytics-zoo/files/analytics-zoo-data/tensorflow_tfauto.tar").!
    "mkdir /tmp/tensorflow_tfauto/".!
    "tar -xvf /tmp/tensorflow_tfauto.tar -C /tmp/tensorflow_tfauto".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/ndarray-128-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString

    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.chwFlag = false
    helper.modelType = "tensorflowSavedModel"
    helper.weightPath = "/tmp/tensorflow_tfauto/"
    ClusterServing.model = helper.loadInferenceModel()

    Seq("sh", "-c", "rm -rf /tmp/tensorflow_tfauto*").!

    val inference = new ClusterServingInference()
    val in = List(("1", b64string, ""), ("2", b64string, ""), ("3", b64string, ""))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 128, "result length wrong")
      require(result(0)._2.length == 1, "result shape wrong")
    })
  }

  "TensorflowModel VGG16" should "work" in {
    ("wget --no-check-certificate -O /tmp/tensorflow_vgg16.tar https://sourceforge.net/" +
"projects/analytics-zoo/files/analytics-zoo-data/tensorflow_vgg16.tar").!
    "mkdir /tmp/tensorflow_vgg16/".!
    "tar -xvf /tmp/tensorflow_vgg16.tar -C /tmp/tensorflow_vgg16".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/image-3_224_224-arrow-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString

    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.chwFlag = false
    helper.modelType = "tensorflowFrozenModel"
    helper.weightPath = "/tmp/tensorflow_vgg16/"
    ClusterServing.model = helper.loadInferenceModel()

    Seq("sh", "-c", "rm -rf /tmp/tensorflow_vgg16*").!

    val inference = new ClusterServingInference()
    val in = List(("1", b64string, ""), ("2", b64string, ""), ("3", b64string, ""))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 1000, "result length wrong")
      require(result(0)._2.length == 1, "result shape wrong")
    })
  }

  "TensorflowModel tf_2out" should "work" in {
    ("wget --no-check-certificate -O /tmp/tensorflow_tf_2out.tar https://sourceforge.net/" +
"projects/analytics-zoo/files/analytics-zoo-data/tensorflow_tf_2out.tar").!
    "mkdir /tmp/tensorflow_tf_2out/".!
    "tar -xvf /tmp/tensorflow_tf_2out.tar -C /tmp/tensorflow_tf_2out".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/ndarray-2-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString

    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.chwFlag = false
    helper.modelType = "tensorflowSavedModel"
    helper.weightPath = "/tmp/tensorflow_tf_2out/"
    ClusterServing.model = helper.loadInferenceModel()

    Seq("sh", "-c", "rm -rf /tmp/tensorflow_tf_2out*").!

    val inference = new ClusterServingInference()
    val in = List(("1", b64string, ""), ("2", b64string, ""), ("3", b64string, ""))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 2, "result length wrong")
      require(result(0)._2.length == 2, "result shape wrong")
    })
  }
  "TF String input" should "work" in {
    ("wget --no-check-certificate -O /tmp/tf_string.tar https://sourceforge.net/" +
"projects/analytics-zoo/files/analytics-zoo-data/tf_string.tar").!
    "tar -xvf /tmp/tf_string.tar -C /tmp/".!

    val model = new InferenceModel(1)
    val modelPath = "/tmp/tf_string"
    model.doLoadTensorflow(modelPath,
      "savedModel", null, null)

    ("rm -rf /tmp/tf_string*").!
    val t = Tensor[String](2)
    t.setValue(1, "123")
    t.setValue(2, "456")
    val res = model.doPredict(t)
    assert(res.toTensor[Float].valueAt(1) == 123)
    assert(res.toTensor[Float].valueAt(2) == 456)
  }
}
