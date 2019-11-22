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
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.common.Utils
import org.scalatest.{BeforeAndAfterAll, FlatSpec, FunSuite, Matchers}

import scala.language.postfixOps
import sys.process._


class PyTorchModelSpec extends FunSuite with Matchers with BeforeAndAfterAll
  with InferenceSupportive {

  val s3Url = "https://s3-ap-southeast-1.amazonaws.com"
  val s3DataUrl = s"$s3Url" + s"/analytics-zoo-models"
  val modelURL = s"$s3DataUrl/pytorch/pytorch-resnet.pt"
  var tmpDir: File = _
  var model: InferenceModel = _
  var model2: InferenceModel = _
  val currentNum = 10
  var modelPath: String = _

  override def beforeAll()  {
    tmpDir = Utils.createTmpDir("ZooVino").toFile()
    val dir = new File(s"${tmpDir.getAbsolutePath}/PyTorchModelSpec").getCanonicalPath
    s"wget -P $dir $modelURL" !;

    s"ls -alh $dir" !;

    modelPath = s"$dir/pytorch-resnet.pt"
    model = new InferenceModel(currentNum) { }
    model2 = new InferenceModel(currentNum) { }
  }

  override def afterAll() {
    model.doRelease()
    model2.doRelease()
    s"rm -rf $tmpDir" !;
  }

  test("pytorch model should be loaded") {
    model.doLoadPyTorch(modelPath)
    println(model)
    model shouldNot be(null)

    val modelBytes = Files.readAllBytes(Paths.get(modelPath))
    model2.doLoadPyTorch(modelBytes)
    println(model2)
    model2 shouldNot be(null)
  }

  test("pytorch model can do predict") {
    val inputTensor = Tensor[Float](1, 3, 224, 224).rand()
    val results = model.doPredict(inputTensor)
    println(results)
    val threads = List.range(0, currentNum).map(i => {
      new Thread() {
        override def run(): Unit = {
          val r = model.doPredict(inputTensor)
          println(r)
          r should be (results)

          val r2 = model2.doPredict(inputTensor)
          println(r2)
          r2 should be (results)
        }
      }
    })
    threads.foreach(_.start())
    threads.foreach(_.join())
  }

}
