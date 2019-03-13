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

import java.io._
import java.util

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.common.CheckedObjectInputStream
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class TestAbstractInferenceModel(supportedConcurrentNum: Integer = 1)
  extends AbstractInferenceModel(supportedConcurrentNum) {
}

class InferenceModelSpec extends FlatSpec with Matchers with BeforeAndAfter
  with InferenceSupportive {
  val resource = getClass().getClassLoader().getResource("models")
  val modelPath = resource.getPath + "/caffe/test_persist.prototxt"
  val weightPath = resource.getPath + "/caffe/test_persist.caffemodel"

  var floatInferenceModel: FloatModel = _

  val inputTensor1 = Tensor[Float](3, 5, 5).rand()
  val inputTensor2 = Tensor[Float](3, 5, 5).rand()
  val inputTensor3 = Tensor[Float](3, 5, 5).rand()
  val inputJTensor1 = transferTensorToJTensor(inputTensor1)
  val inputJTensor2 = transferTensorToJTensor(inputTensor2)
  val inputJTensor3 = transferTensorToJTensor(inputTensor3)
  val inputTensorsArray = Array(inputJTensor1, inputJTensor2, inputJTensor3)

  val inputTensorList1 = util.Arrays.asList(inputJTensor1)
  val inputTensorList2 = util.Arrays.asList(inputJTensor2)
  val inputTensorList3 = util.Arrays.asList(inputJTensor3)
  val inputTensorList = util.Arrays.asList(inputTensorList1, inputTensorList2, inputTensorList3)

  before {
    floatInferenceModel = InferenceModelFactory.loadFloatModelForCaffe(modelPath, weightPath)
  }

  after {
    System.clearProperty("bigdl.localMode")
    System.clearProperty("bigdl.coreNumber")
  }

  "InferenceModel" should "load as time reduced, weights shared, space reduced" in {
    val supportedConcurrentNum = 100
    val aModel = new InferenceModel(supportedConcurrentNum)
    val begin1 = System.currentTimeMillis()
    aModel.doLoadCaffe(modelPath, weightPath)
    val end1 = System.currentTimeMillis()
    val time1 = end1 - begin1

    val begin2 = System.currentTimeMillis()
    val fModels = List.range(0, supportedConcurrentNum).map(i => InferenceModelFactory.
      loadFloatModelForCaffe(modelPath, weightPath))
    val end2 = System.currentTimeMillis()
    val time2 = end2 - begin2

    println(s"load $supportedConcurrentNum shared wights models used $time1 ms," +
      s"load $supportedConcurrentNum single models used $time2 ms.")

    val weightsForAModel1 = aModel.modelQueue.take().asInstanceOf[FloatModel].
      model.getWeightsBias()(0).storage()
    val weightsForAModel2 = aModel.modelQueue.take().asInstanceOf[FloatModel].
      model.getWeightsBias()(0).storage()
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
    val aModel2 = in4AModel.readObject.asInstanceOf[InferenceModel]
    in4AModel.close()

    val weightsForAModel3 = aModel2.modelQueue.take().asInstanceOf[FloatModel]
      .model.getWeightsBias()(0).storage()
    val weightsForAModel4 = aModel2.modelQueue.take().asInstanceOf[FloatModel]
      .model.getWeightsBias()(0).storage()
    assert(weightsForAModel3 == weightsForAModel4)

    val inputTensor = Tensor[Float](3, 5, 5).rand()
    val result1 = fModels(0).predict(inputTensor)
    result1.toTensor[Float].size() should be(Array(2))

    val inputTensorBatch = Tensor[Float](5, 3, 5, 5).rand()
    val result2 = fModels(0).predict(inputTensorBatch)
    result2.toTensor[Float].size() should be(Array(5, 2))

    val currentNum = 1000
    val begin3 = System.currentTimeMillis()
    val threads = List.range(0, currentNum).map(i => {
      new Thread() {
        override def run(): Unit = {
          val r1 = aModel2.doPredict(inputTensorBatch)
          r1.toTensor[Float].size() should be(Array(5, 2))
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
          val r1 = fModels(0).predict(inputTensorBatch)
          r1.toTensor[Float].size() should be(Array(5, 2))
        }
      }
    })
    threads2.foreach(_.start())
    threads2.foreach(_.join())
    val end4 = System.currentTimeMillis()
    val time4 = end4 - begin4
    println(s"$currentNum concurrently predict used: $time3 ms," +
      s"$currentNum concurrently predict used: $time4 ms.")
  }


  "AbstractInferenceModel" should "load as time reduced, weights shared, space reduced" in {
    val supportedConcurrentNum = 100
    val aModel = new TestAbstractInferenceModel(supportedConcurrentNum)
    val begin1 = System.currentTimeMillis()
    aModel.loadCaffe(modelPath, weightPath)
    val end1 = System.currentTimeMillis()
    val time1 = end1 - begin1

    val begin2 = System.currentTimeMillis()
    val fModels = List.range(0, supportedConcurrentNum).map(i => InferenceModelFactory.
      loadFloatModelForCaffe(modelPath, weightPath))
    val end2 = System.currentTimeMillis()
    val time2 = end2 - begin2

    println(s"load $supportedConcurrentNum shared wights models used $time1 ms," +
      s"load $supportedConcurrentNum single models used $time2 ms.")

    val weightsForAModel1 = aModel.modelQueue.take().asInstanceOf[FloatModel]
      .model.getWeightsBias()(0).storage()
    val weightsForAModel2 = aModel.modelQueue.take().asInstanceOf[FloatModel]
      .model.getWeightsBias()(0).storage()
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
    val in4AModel = new CheckedObjectInputStream(classOf[TestAbstractInferenceModel], bis4AModel)
    val aModel2 = in4AModel.readObject.asInstanceOf[TestAbstractInferenceModel]
    in4AModel.close()

    val weightsForAModel3 = aModel2.modelQueue.take().asInstanceOf[FloatModel]
      .model.getWeightsBias()(0).storage()
    val weightsForAModel4 = aModel2.modelQueue.take().asInstanceOf[FloatModel]
      .model.getWeightsBias()(0).storage()
    assert(weightsForAModel3 == weightsForAModel4)

    val currentNum = 10
    val begin3 = System.currentTimeMillis()
    val threads = List.range(0, currentNum).map(i => {
      new Thread() {
        override def run(): Unit = {
          val r1 = aModel2.predict(inputTensorList)
          r1.size() should be(3)
          r1.get(0).get(0).getShape should be(Array(2))
        }
      }
    })
    threads.foreach(_.start())
    threads.foreach(_.join())
    val end3 = System.currentTimeMillis()
    val time3 = end3 - begin3
  }

  "JTensor toString" should "return element" in {
    val data = Array(1.0f, 2.0f, 3.0f, 4.0f)
    val shape = Array(1, 4)
    val jTensor = new JTensor(data, shape)
    jTensor.toString should be("JTensor{data=[1.0, 2.0, 3.0, 4.0], shape=[1, 4]}")
  }
}
