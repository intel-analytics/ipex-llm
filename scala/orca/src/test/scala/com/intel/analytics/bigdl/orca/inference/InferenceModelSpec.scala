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

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}
import java.util

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class TestAbstractInferenceModel(supportedConcurrentNum: Integer = 1)
  extends AbstractInferenceModel(supportedConcurrentNum) {
}

class InferenceModelSpec extends FlatSpec with Matchers with BeforeAndAfter
  with InferenceSupportive {
  val resource = getClass().getClassLoader().getResource("models")
  val modelPath = resource.getPath + "/caffe/test_persist.prototxt"
  val weightPath = resource.getPath + "/caffe/test_persist.caffemodel"

  var floatInferenceModel: FloatInferenceModel = _
  // var abstractInferenceModel = new AbstractInferenceModel(1000) {}

  before {
    floatInferenceModel = InferenceModelFactory.
      loadFloatInferenceModelForCaffe(modelPath, weightPath)
  }

  after {
    System.clearProperty("bigdl.localMode")
    System.clearProperty("bigdl.coreNumber")
  }

  "AbstractInferenceModel" should "load as time reduced, weights shared, space reduced" in {
    val supportedConcurrentNum = 10
    val aModel = new TestAbstractInferenceModel(supportedConcurrentNum)
    val begin1 = System.currentTimeMillis()
    aModel.loadCaffe(modelPath, weightPath)
    val end1 = System.currentTimeMillis()
    val time1 = end1 - begin1

    val begin2 = System.currentTimeMillis()
    val fModels = List.range(0, supportedConcurrentNum).map(i => InferenceModelFactory.
      loadFloatInferenceModelForCaffe(modelPath, weightPath))
    val end2 = System.currentTimeMillis()
    val time2 = end2 - begin2

    println(s"load $supportedConcurrentNum shared wights models used $time1 ms," +
      s"load $supportedConcurrentNum single models used $time2 ms.")

    val weightsForAModel1 = aModel.modelQueue.take().model.getWeightsBias()(0).storage()
    val weightsForAModel2 = aModel.modelQueue.take().model.getWeightsBias()(0).storage()
    assert(weightsForAModel1 == weightsForAModel2)

    val weightsForFModel1 = fModels(0).model.getWeightsBias()(0).storage()
    val weightsForFModel2 = fModels(1).model.getWeightsBias()(0).storage()
    assert(weightsForFModel1 != weightsForFModel2)

    val bos4AModel = new ByteArrayOutputStream
    val out4AModel = new ObjectOutputStream(bos4AModel)
    out4AModel.writeObject(aModel)
    out4AModel.flush()
    val bytes4AModel = bos4AModel.toByteArray()
    bos4AModel.close()

    val bos4FModel = new ByteArrayOutputStream
    val out4FModel = new ObjectOutputStream(bos4FModel)
    out4FModel.writeObject(fModels)
    out4FModel.flush()
    val bytes4FModel = bos4FModel.toByteArray()
    bos4FModel.close()
    println(s"load $supportedConcurrentNum shared wights models bytes: ${bytes4AModel.length}," +
      s"load $supportedConcurrentNum single models bytes: ${bytes4FModel.length}.")
    assert(bytes4AModel.length < bytes4FModel.length)

    val bis4AModel = new ByteArrayInputStream(bytes4AModel)
    val in4AModel = new ObjectInputStream(bis4AModel)
    val aModel2 = in4AModel.readObject.asInstanceOf[TestAbstractInferenceModel]
    in4AModel.close()

    val weightsForAModel3 = aModel2.modelQueue.take().model.getWeightsBias()(0).storage()
    val weightsForAModel4 = aModel2.modelQueue.take().model.getWeightsBias()(0).storage()
    assert(weightsForAModel3 == weightsForAModel4)

    val inputTensor = Tensor[Float](3, 5, 5).rand()
    val input = new util.ArrayList[JTensor]()
    input.add(transferTensorToJTensor(inputTensor))
    val inputs = new util.ArrayList[util.List[JTensor]]()
    inputs.add(input)
    val result = fModels(0).predict(inputs).get(0).get(0)
    val data = result.getData.mkString(",")
    val shape = result.getShape.mkString(",")

    val currentNum = 10
    val begin3 = System.currentTimeMillis()
    val threads = List.range(0, currentNum).map(i => {
      new Thread() {
        override def run(): Unit = {
          val r = aModel2.predict(inputs).get(0).get(0)
          val d = r.getData.mkString(",")
          val s = r.getShape.mkString(",")
          assert(d == data)
          assert(s == shape)
        }
      }
    })
    threads.foreach(_.start())
    threads.foreach(_.join())
    val end3 = System.currentTimeMillis()
    val time3 = end3 - begin3

    val begin4 = System.currentTimeMillis()
    val threads2 = List.range(0, currentNum).map(i => {
      new Thread() {
        override def run(): Unit = {
          val r = fModels(0).predict(inputs).get(0).get(0)
          val d = r.getData.mkString(",")
          val s = r.getShape.mkString(",")
        }
      }
    })
    threads2.foreach(_.start())
    threads2.foreach(_.join())
    val end4 = System.currentTimeMillis()
    val time4 = end4 - begin4
  }


  "model " should "serialize" in {
    val bos = new ByteArrayOutputStream
    val out = new ObjectOutputStream(bos)
    out.writeObject(floatInferenceModel)
    out.flush()
    val bytes = bos.toByteArray()
    bos.close()

    val bis = new ByteArrayInputStream(bytes)
    val in = new ObjectInputStream(bis)
    val floatInferenceModel2 = in.readObject.asInstanceOf[FloatInferenceModel]
    // println(floatInferenceModel2.predictor)
    assert(floatInferenceModel.model == floatInferenceModel2.model)
    in.close()
  }

  "JTensor toString" should "return element" in {
    val data = Array(1.0f, 2.0f, 3.0f, 4.0f)
    val shape = Array(1, 4)
    val jTensor = new JTensor(data, shape)
    jTensor.toString should be ("JTensor{data=[1.0, 2.0, 3.0, 4.0], shape=[1, 4]}")
  }
}
