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
package com.intel.analytics.bigdl.utils

import java.nio.file.Paths

import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.models.resnet.Convolution
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericDouble
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.caffe.{CaffeConversionException, CaffeLoader, CaffePersister, Customizable}

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random


class CaffePersisterSpec extends FlatSpec with Matchers{

  val resource = getClass().getClassLoader().getResource("caffe")
  val prototxt = Paths.get(TestUtils.processPath(resource.getPath()), "test.prototxt").toString
  val modelPath = Paths.get(TestUtils.processPath(resource.getPath()), "test.caffemodel").toString

  val savedprototxt = Paths.get(TestUtils.processPath(resource.getPath()),
    "test_persist.prototxt").toString
  val savedmodelPath = Paths.get(TestUtils.processPath(resource.getPath()),
    "test_persist.caffemodel").toString

  private class LoadDummy[T: ClassTag](implicit ev: TensorNumeric[T]) extends Customizable[T] {
    override def convertor(layer: GeneratedMessage): Seq[ModuleNode[T]] = {
      Seq(Identity[T].setName("Dummy").inputs())
    }
  }

  val convertMap = new mutable.HashMap[String, Customizable[Double]]()
  convertMap("DUMMY") = new LoadDummy[Double]

  "Save graph module" should "Works properly" in {

    val convolution1 = new ModuleNode(Convolution(3, 4, 2, 2).
      setName("conv1").asInstanceOf[AbstractModule[Activity, Activity, Double]])

    val convolution2 = new ModuleNode(Convolution(4, 3, 2, 2).setName("conv2")
      .asInstanceOf[AbstractModule[Activity, Activity, Double]])

    val view = new ModuleNode(View(27).setName("view")
      .asInstanceOf[AbstractModule[Activity, Activity, Double]])

    val ip = new ModuleNode(Linear(2, 27, withBias = false).setName("ip")
      .asInstanceOf[AbstractModule[Activity, Activity, Double]])

    convolution1 -> convolution2

    convolution2 -> view

    view -> ip

    val module = Graph(convolution1, ip)

    CaffePersister.persist("/tmp/test.prototxt", "/tmp/test.caffemodel", module, overwrite = true)
  }

  "Persist V1 module" should "works properly" in {
    val module = new ModuleNode(Linear[Double](100, 10).
      setName("simple linear").asInstanceOf[AbstractModule[Activity, Activity, Double]])
    val graph = Graph(module, module)
    CaffePersister.persist("/tmp/v1.prototxt", "/tmp/v1.caffemodel",
      graph, useV2 = false, overwrite = true)
  }

  "Persist V2 module" should "works properly" in {
    val module = new ModuleNode(Linear[Double](100, 10).
      setName("simple linear").asInstanceOf[AbstractModule[Activity, Activity, Double]])
    val graph = Graph(module, module)
    CaffePersister.persist("/tmp/v2.prototxt", "/tmp/v2.caffemodel",
      graph, useV2 = true, overwrite = true)
  }

  "A saved module" should "have same result as pre-saved one" in {

    val input1 = Tensor[Double](1, 3, 5, 5).apply1( e => Random.nextDouble())
    val input2 = Tensor[Double]()
    input2.resizeAs(input1).copy(input1)

    val preSavedModule = CaffeLoader.loadCaffe(prototxt, modelPath,
      customizedConverters = convertMap)._1

    val savedModule = CaffeLoader.loadCaffe(savedprototxt, savedmodelPath)._1

    val preSavedResult = preSavedModule.forward(input1)

    val savedResult = savedModule.forward(input2)

    preSavedResult should be (savedResult)
  }

  "Save sequantial module" should "throw exception" in {
    val module = Sequential()
      .add(Convolution(3, 4, 2, 2).setName("conv"))
      .add(Convolution(4, 3, 2, 2).setName("conv2"))
      .add(View(27)).setName("view")
      .add(Linear(2, 27, withBias = false).setName("ip"))
    intercept[CaffeConversionException] {
      CaffePersister.persist("/tmp/test.prototxt", "/tmp/test.caffemodel", module)
    }

  }

}
