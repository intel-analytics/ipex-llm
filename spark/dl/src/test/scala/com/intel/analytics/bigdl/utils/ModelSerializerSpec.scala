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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericDouble
import com.intel.analytics.bigdl.utils.serialization.ModelSerializer
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class ModelSerializerSpec extends FlatSpec with Matchers {

  "Abs serializer" should "work properly" in {
    val abs = Abs[Double]()
    val tensor1 = Tensor[Double](10).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    val res1 = abs.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModelSerializer.saveToFile("/tmp/abs.bigdl", abs, true)
    val loadedAbs = ModelSerializer.loadFromFile("/tmp/abs.bigdl").asInstanceOf[Abs[Double]]
    val res2 = loadedAbs.forward(tensor2)
    res1 should be (res2)
  }

  "Add serializer" should "work properly" in {
    val add = Add[Double](5)
    val tensor1 = Tensor[Double](5).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    val res1 = add.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModelSerializer.saveToFile("/tmp/add.bigdl", add, true)
    val loadedAdd = ModelSerializer.loadFromFile("/tmp/add.bigdl")
    val res2 = loadedAdd.asInstanceOf[Add[Double]].forward(tensor2)
    res1 should be (res2)
  }

  "Linear serializer" should "work properly" in {
    val linear = Linear[Double](10, 2)
    val tensor1 = Tensor[Double](10).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    val res1 = linear.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModelSerializer.saveToFile("/tmp/linear.bigdl", linear, true)
    val loadedLinear = ModelSerializer.loadFromFile("/tmp/linear.bigdl")
    val res2 = loadedLinear.asInstanceOf[Linear[Double]].forward(tensor2)
    res1 should be (res2)
   }

  "Sequantial Container" should "work properly" in {
    val sequential = Sequential[Double]()
    val linear = Linear[Double](10, 2)
    sequential.add(linear)
    val tensor1 = Tensor[Double](10).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    val res1 = sequential.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModelSerializer.saveToFile("/tmp/sequential.bigdl", sequential, true)
    val loadedSequential = ModelSerializer.loadFromFile("/tmp/sequential.bigdl")
    val res2 = loadedSequential.asInstanceOf[Sequential[Double]].forward(tensor2)
    res1 should be (res2)
  }

  "Graph Container" should "work properly" in {
    val linear = Linear[Double](10, 2).apply()
    val graph = Graph(linear, linear)
    val tensor1 = Tensor[Double](10).apply1(_ => Random.nextDouble())
    val tensor2 = Tensor[Double]()
    val res1 = graph.forward(tensor1)
    tensor2.resizeAs(tensor1).copy(tensor1)
    ModelSerializer.saveToFile("/tmp/graph.bigdl", graph, true)
    val loadedGraph = ModelSerializer.loadFromFile("/tmp/graph.bigdl")
    val res2 = loadedGraph.asInstanceOf[Graph[Double]].forward(tensor2)
    res1 should be (res2)
  }
}
